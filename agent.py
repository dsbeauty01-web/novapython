"""
Nova v200 — Full LiveKit Agent worker (Days 1-5 complete)

This is the BRAIN. It owns every word Nova says.

Architecture:
    Kid voice ──► Deepgram STT ──► Claude (phase-aware prompt) ──► ElevenLabs TTS
                                                                       │
                                                                       ▼
                                                              Runway plugin
                                                                       │
                                                                       ▼
                                                              Nova's face lipsync

Key design decisions:
- Phase switching: recognition → dance → goodbye (via room data messages)
- 2.5s minimum pacing baked in: PaceGate enforces it
- Escalation: streak-based tier (soft → warm → big) picked by personality module
- Memory: in-memory MemoryStore per kid_id (across sessions within uptime)
- Vision: Gemini Flash for "she sees me" — fires once per session at natural moment
"""

import os
import json
import time
import asyncio
import logging
from typing import Optional

from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import (
    deepgram,
    anthropic as anthropic_plugin,
    elevenlabs,
    runway,
    silero,
)

# Optional: turn detector for natural conversation pauses
try:
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
    TURN_DETECTOR_AVAILABLE = True
except ImportError:
    TURN_DETECTOR_AVAILABLE = False

import personality
import memory
import vision

load_dotenv()

logger = logging.getLogger("nova-v200")
logging.basicConfig(level=logging.INFO)


# ────────────────────────────────────────────────────────────────────────
# Pacing gate — keeps Nova from talking over herself.
#
# PRIMARY (smart): wait until Nova has actually FINISHED her last line.
#   We watch the agent's speaking state; when she stops speaking, she's free
#   to speak again. This is dynamic — as fast as 0s, as long as her sentence.
#
# BACKUP (safety net): a small minimum gap, in case the "done speaking" signal
#   never arrives. Tunable via the NOVA_MIN_GAP_SEC environment variable.
#   (The old hardcoded 2.5s was a workaround for the previous architecture
#    where Runway ran its own brain. That setup is gone — Runway is now just a
#    mouth — so the floor can be much lower.)
# ────────────────────────────────────────────────────────────────────────
class PaceGate:
    """Lets Nova speak again once she's done — with a small safety floor."""

    # Backup floor, in seconds. Dial it down at test time without code edits.
    MIN_GAP_SEC = float(os.getenv("NOVA_MIN_GAP_SEC", "1.2"))

    def __init__(self):
        self._last_spoke = 0.0
        self._lock = asyncio.Lock()
        self._is_speaking = False  # flipped by agent speaking-state events

    def mark_speaking(self, speaking: bool):
        """Called from the agent's state-changed event."""
        self._is_speaking = speaking
        if not speaking:
            self._last_spoke = time.time()

    async def acquire(self):
        """Wait until Nova is free to speak: not mid-speech, and past the floor."""
        async with self._lock:
            # PRIMARY: if she's still talking, wait for her to finish.
            # Poll briefly — the speaking flag flips off on AgentSpeechEnded.
            waited = 0.0
            while self._is_speaking and waited < 8.0:  # 8s hard ceiling, never hang
                await asyncio.sleep(0.05)
                waited += 0.05

            # BACKUP: enforce the small safety floor since her last line ended.
            now = time.time()
            gap = now - self._last_spoke
            if gap < self.MIN_GAP_SEC:
                wait = self.MIN_GAP_SEC - gap
                logger.info(f"[pace] backup floor: waiting {wait:.2f}s")
                await asyncio.sleep(wait)
            self._last_spoke = time.time()


# ────────────────────────────────────────────────────────────────────────
# Session state — held in agent instance, mutated as game progresses
# ────────────────────────────────────────────────────────────────────────
class NovaSessionState:
    """Per-room state Nova tracks for THIS conversation."""

    def __init__(self, kid_id: Optional[str] = None):
        self.kid_id = kid_id or f"anon-{int(time.time())}"
        self.ctx = personality.NovaContext(phase="recognition")
        self.pace = PaceGate()
        self.vision_fired = False
        self.greeting_done = False
        self.session_started_at = time.time()
        # If we have memory for this kid, prefill the context
        mem = memory.store.get(self.kid_id)
        if mem.name:
            self.ctx.name = mem.name
            self.ctx.sessions_before = mem.total_sessions
            self.ctx.max_streak = mem.max_streak
            self.ctx.favorite_move = mem.favorite_move
            if mem.best_moments:
                self.ctx.best_moment = mem.best_moments[-1]

    def push_event(self, event: dict):
        """Browser pushed a game event."""
        ev = event.get("event")
        if not ev:
            return

        # Phase transitions
        if ev == "phase":
            new_phase = event.get("phase")
            if new_phase in ("recognition", "dance", "goodbye"):
                logger.info(f"[state] phase {self.ctx.phase} → {new_phase}")
                self.ctx.phase = new_phase

        # Game events
        elif ev == "hit":
            self.ctx.hits = event.get("hits", self.ctx.hits + 1)
            self.ctx.streak = event.get("streak", self.ctx.streak + 1)
            self.ctx.last_event = "hit"
            # Track today's best streak too, so goodbye celebrates THIS session
            if self.ctx.streak > self.ctx.max_streak:
                self.ctx.max_streak = self.ctx.streak
            memory.store.record_streak(self.kid_id, self.ctx.streak)

        elif ev == "miss":
            self.ctx.streak = 0
            self.ctx.last_event = "miss"

        elif ev == "first_hit":
            self.ctx.hits = 1
            self.ctx.streak = 1
            self.ctx.last_event = "first_hit"

        elif ev == "freeze_hit":
            self.ctx.last_event = "freeze_hit"

        elif ev == "freeze_miss":
            self.ctx.last_event = "freeze_miss"
            self.ctx.streak = 0

        elif ev == "music_tick":
            self.ctx.music_sec = float(event.get("sec", 0))

        elif ev == "name":
            new_name = event.get("name", "").strip()
            if new_name:
                self.ctx.name = new_name
                memory.store.update(self.kid_id, name=new_name)
                logger.info(f"[state] name captured: {new_name}")

        elif ev == "best_moment":
            moment = event.get("moment", "").strip()
            if moment:
                memory.store.add_moment(self.kid_id, moment)
                self.ctx.best_moment = moment

        elif ev == "vision":
            obs = event.get("observation", "").strip()
            if obs:
                self.ctx.observed_visual = obs

    def system_prompt(self) -> str:
        """Build the prompt for THIS moment."""
        return personality.build_system_prompt(self.ctx)


# ────────────────────────────────────────────────────────────────────────
# Nova Agent — the LiveKit Agent subclass
# ────────────────────────────────────────────────────────────────────────
class NovaAgent(Agent):
    def __init__(self, state: NovaSessionState):
        super().__init__(instructions=state.system_prompt())
        self.state = state

    def refresh_instructions(self):
        """Rebuild instructions based on current state (called each turn)."""
        new_prompt = self.state.system_prompt()
        self.update_instructions(new_prompt)

    async def on_user_turn_completed(self, chat_ctx, new_message):
        """Hook fired when kid finishes speaking. Refresh prompt + pace."""
        self.refresh_instructions()
        await self.state.pace.acquire()


# ────────────────────────────────────────────────────────────────────────
# Per-room control channel — browser pushes game events via LiveKit data
# ────────────────────────────────────────────────────────────────────────
def register_data_handler(room: rtc.Room, state: NovaSessionState, session: AgentSession):
    """Listen for game events from the browser."""

    @room.on("data_received")
    def on_data(packet: rtc.DataPacket):
        try:
            raw = packet.data.decode("utf-8")
            msg = json.loads(raw)
            kind = msg.get("kind")

            if kind == "game-event":
                event = msg.get("event", {})
                logger.info(f"[data] game-event: {event}")
                state.push_event(event)

                # If a hit/miss happened during DANCE, react immediately
                if state.ctx.phase == "dance" and event.get("event") in (
                    "hit", "miss", "first_hit", "freeze_hit", "freeze_miss"
                ):
                    asyncio.create_task(_react_to_event(session, state))

                # Phase transition to goodbye → speak final goodbye
                if event.get("event") == "phase" and event.get("phase") == "goodbye":
                    asyncio.create_task(_speak_goodbye(session, state))

            elif kind == "vision-observation":
                obs = msg.get("text", "").strip()
                if obs and not state.vision_fired:
                    state.vision_fired = True
                    state.ctx.observed_visual = obs
                    logger.info(f"[data] vision observation: '{obs}'")
                    asyncio.create_task(_drop_in_observation(session, state, obs))

        except Exception as e:
            logger.error(f"[data] parse error: {e}")


# ────────────────────────────────────────────────────────────────────────
# Reaction helpers
# ────────────────────────────────────────────────────────────────────────
async def _react_to_event(session: AgentSession, state: NovaSessionState):
    """Game event happened — generate phase-aware reaction."""
    await state.pace.acquire()
    # Refresh prompt to current phase
    if hasattr(session.agent, "refresh_instructions"):
        session.agent.refresh_instructions()
    instructions = (
        f"React to game event '{state.ctx.last_event}'. "
        f"Current streak: {state.ctx.streak}. "
        f"Reply 1-6 words only. Follow your dance phase rules."
    )
    try:
        await session.generate_reply(instructions=instructions)
    except Exception as e:
        logger.error(f"[react] generate_reply failed: {e}")


async def _speak_goodbye(session: AgentSession, state: NovaSessionState):
    """Phase transitioned to goodbye — warm wrap-up."""
    await state.pace.acquire()
    memory.store.increment_sessions(state.kid_id)
    if hasattr(session.agent, "refresh_instructions"):
        session.agent.refresh_instructions()
    instructions = (
        f"The song ended. Speak warm goodbye now. "
        f"Stats: hits={state.ctx.hits}, max_streak={state.ctx.max_streak}. "
        f"Follow goodbye phase rules: ONE specific celebration + ONE open question."
    )
    try:
        await session.generate_reply(instructions=instructions)
    except Exception as e:
        logger.error(f"[goodbye] generate_reply failed: {e}")


async def _drop_in_observation(session: AgentSession, state: NovaSessionState, observation: str):
    """Drop in vision observation naturally."""
    await state.pace.acquire()
    if hasattr(session.agent, "refresh_instructions"):
        session.agent.refresh_instructions()
    instructions = (
        f"You just noticed: '{observation}'. "
        f"Say it warmly with '...' pauses, like you spotted it. ONE sentence."
    )
    try:
        await session.generate_reply(instructions=instructions)
    except Exception as e:
        logger.error(f"[vision] generate_reply failed: {e}")


# ────────────────────────────────────────────────────────────────────────
# Background: gentle idle engagement
#
# If the child goes quiet for a while, calm Nova softly stays present —
# ONE gentle line, then a long cooldown so she never nags. She NEVER changes
# phase from here (that was an old v113 bug) and NEVER says "are you there?".
#
# Timing is tunable at test time:
#   NOVA_IDLE_SILENCE_SEC  — how long quiet before a nudge (default 14s)
#   NOVA_IDLE_COOLDOWN_SEC — min gap between nudges      (default 25s)
# ────────────────────────────────────────────────────────────────────────
import random

async def _idle_watch_loop(session: AgentSession, state: NovaSessionState):
    """Watch for long silence; offer ONE soft line, rarely."""
    silence_limit = float(os.getenv("NOVA_IDLE_SILENCE_SEC", "14"))
    cooldown = float(os.getenv("NOVA_IDLE_COOLDOWN_SEC", "25"))
    last_nudge = 0.0

    while True:
        await asyncio.sleep(2)  # check every couple seconds

        # Never nudge while Nova is mid-speech, or during goodbye wrap-up edge
        if state.pace._is_speaking:
            continue

        now = time.time()
        quiet_for = now - state.pace._last_spoke
        since_last_nudge = now - last_nudge

        if quiet_for < silence_limit or since_last_nudge < cooldown:
            continue

        # Pick a soft line for the CURRENT phase (no phase change, ever)
        phase = state.ctx.phase
        bank_key = {
            "recognition": "idle_recognition",
            "dance": "idle_dance",
            "goodbye": "idle_goodbye",
        }.get(phase, "idle_recognition")
        lines = personality.PHRASE_BANKS.get(bank_key, ["mhm... I'm here..."])
        line = random.choice(lines)

        try:
            await state.pace.acquire()
            await session.say(line)
            last_nudge = time.time()
            logger.info(f"[idle] soft nudge ({phase}): '{line}'")
        except Exception as e:
            logger.error(f"[idle] nudge failed: {e}")


# ────────────────────────────────────────────────────────────────────────
# Background: trigger vision once at the natural moment
# ────────────────────────────────────────────────────────────────────────
async def _vision_trigger_loop(room: rtc.Room, state: NovaSessionState):
    """Ask browser for a webcam frame ~10s into recognition."""
    await asyncio.sleep(10)
    if state.vision_fired or state.ctx.phase != "recognition":
        return
    try:
        msg = json.dumps({"kind": "request-vision"})
        await room.local_participant.publish_data(msg.encode("utf-8"), reliable=True)
        logger.info("[vision] requested webcam frame from browser")
    except Exception as e:
        logger.error(f"[vision] request failed: {e}")


# ────────────────────────────────────────────────────────────────────────
# Entrypoint — once per Runway session
# ────────────────────────────────────────────────────────────────────────
async def entrypoint(ctx: JobContext):
    logger.info(f"[nova-v200] entrypoint room={ctx.room.name}")

    kid_id = None
    try:
        if ctx.room.metadata:
            meta = json.loads(ctx.room.metadata)
            kid_id = meta.get("kidId")
    except Exception:
        pass

    state = NovaSessionState(kid_id=kid_id)
    logger.info(
        f"[nova-v200] kid_id={state.kid_id} "
        f"name={state.ctx.name} sessions_before={state.ctx.sessions_before}"
    )

    avatar_id = os.getenv("NOVA_AVATAR_ID", "e976bbb2-de60-4da6-845e-4b754050e55b")

    # Build session pipeline
    session_kwargs = dict(
        stt=deepgram.STT(
            model="nova-3",
            language="multi",
            interim_results=True,
        ),
        llm=anthropic_plugin.LLM(
            model="claude-haiku-4-5-20251001",
            temperature=0.7,
        ),
        tts=elevenlabs.TTS(
            # Matilda — young, warm female ("gentle big sister"). Swap by ear:
            # Charlotte (soft/breathy), Dorothy (kids' stories), Lily (warm young).
            # Override at test time with NOVA_VOICE_ID — no code edit needed.
            voice_id=os.getenv("NOVA_VOICE_ID", "XrExE9yKIg1WjnnlVkGX"),
            model="eleven_flash_v2_5",
            voice_settings=elevenlabs.VoiceSettings(
                # Calm but ALIVE — not flat. Tunable via env for test-time tuning.
                stability=float(os.getenv("NOVA_VOICE_STABILITY", "0.5")),
                similarity_boost=float(os.getenv("NOVA_VOICE_SIMILARITY", "0.75")),
                style=float(os.getenv("NOVA_VOICE_STYLE", "0.2")),
                use_speaker_boost=True,
            ),
        ),
        vad=silero.VAD.load(),
    )
    # Turn-detector disabled: its model file (model_q8.onnx) wasn't available
    # in this environment. Silero VAD (already configured) handles turn-taking
    # just fine. Re-enable later if you choose to bake the model into the build.
    if False:  # was: if TURN_DETECTOR_AVAILABLE:
        session_kwargs["turn_detection"] = MultilingualModel()
        logger.info("[nova-v200] turn detector enabled")

    session = AgentSession(**session_kwargs)
    logger.info("[nova-v200] step 1: AgentSession created")

    # Runway face plugin
    try:
        runway_avatar = runway.AvatarSession(avatar_id=avatar_id)
        await runway_avatar.start(session, room=ctx.room)
        logger.info(f"[nova-v200] step 2: runway avatar started, id={avatar_id[:8]}")
    except Exception as e:
        logger.exception(f"[nova-v200] CRASH at runway start: {e}")
        raise

    # The agent
    agent = NovaAgent(state)
    logger.info("[nova-v200] step 3: NovaAgent created")

    # Data channel listener BEFORE session starts (catch early events)
    register_data_handler(ctx.room, state, session)
    logger.info("[nova-v200] step 4: data handler registered")

    try:
        await session.start(
            agent=agent,
            room=ctx.room,
        )
        logger.info("[nova-v200] step 5: session.start COMPLETE")
    except Exception as e:
        logger.exception(f"[nova-v200] CRASH at session.start: {e}")
        raise

    # Connect Nova's speaking state to the PaceGate (smart pacing).
    try:
        @session.on("agent_state_changed")
        def _on_agent_state(ev):
            speaking = getattr(ev, "new_state", None) == "speaking"
            state.pace.mark_speaking(speaking)
        logger.info("[nova-v200] step 6: smart pacing hooked")
    except Exception as e:
        logger.warning(f"[nova-v200] smart pacing unavailable: {e}")

    # GREETING — first words from OUR brain
    state.greeting_done = True
    logger.info("[nova-v200] step 7: about to generate greeting...")
    try:
        if state.ctx.name and state.ctx.sessions_before > 0:
            await session.generate_reply(
                instructions=(
                    f"Greet {state.ctx.name} warmly — they came back. "
                    f"Use their name. ONE sentence with '...' pauses."
                )
            )
        else:
            await session.generate_reply(
                instructions=(
                    "Greet kid softly. Ask their name. "
                    "Say something like: 'oh hi friend... I'm Nova... what's your name?'"
                )
            )
        logger.info("[nova-v200] step 8: GREETING SENT SUCCESSFULLY")
    except Exception as e:
        logger.exception(f"[nova-v200] CRASH at greeting: {e}")
        raise

    # Kick off vision request in background
    asyncio.create_task(_vision_trigger_loop(ctx.room, state))

    # Gentle idle engagement — soft presence if the child goes quiet
    asyncio.create_task(_idle_watch_loop(session, state))


# ────────────────────────────────────────────────────────────────────────
# Worker boot
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            # Named agent → server explicitly dispatches "nova" into each room.
            # Without this, newer LiveKit won't auto-route the worker to rooms.
            agent_name="nova",
        )
    )
