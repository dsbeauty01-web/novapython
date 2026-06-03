"""
Nova v201 — LiveKit Agent worker (OpenAI brain · ElevenLabs Lily voice · Runway face)

This is Nova's BRAIN. It owns every word she says.

Architecture (Jun 3 2026):
    Kid voice ──► Deepgram STT ──► OpenAI gpt-4o-mini ──► ElevenLabs Lily TTS
                                       (phase-aware prompt)            │
                                                                       ▼
                                                              Runway plugin
                                                                       │
                                                                       ▼
                                                              Nova's face lipsync

Character: Nova is a warm American ~20yo dance friend. Cool-older-cousin energy.
Soul lives in personality.py — three phase prompts (recognition / dance / goodbye).

History of brain choices:
- v113: Runway's hidden brain (broke character, abandoned)
- v200 early: Anthropic Claude Haiku (timed out repeatedly under load)
- v201 mid:   Gemini 2.5 Flash (blocked our fairy greeting as PROHIBITED_CONTENT
              — Google's kids-safety classifier is too aggressive)
- v201 now:   OpenAI gpt-4o-mini (industry-standard for kids apps: Loora, Speak)

Key design decisions:
- Phase switching: recognition → dance → goodbye (browser pushes via data channel)
- Pacing: PaceGate keeps Nova from talking over herself (1.2s floor, smart wait)
- Memory: in-memory MemoryStore per kid_id (within uptime; Postgres later)
- Vision: Gemini Flash (separate path) for "she sees me" moment
- Heavy logging: [HEAR]/[TYPE]/[BRAIN]/[SPEAK]/[PACKET]/[MIC-IN] tags so we
  can pinpoint exactly where a session breaks.
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
    elevenlabs,
    runway,
    silero,
)

# Brain = OpenAI (industry standard for kids apps — Loora, Speak, etc.).
# Gemini is gone from the brain (its safety filters blocked our fairy prompt
# as PROHIBITED_CONTENT — verified in production logs Jun 3 2026).
# Gemini is still used for VISION via the separate google-generativeai client,
# because vision works fine — only the LLM was getting blocked.
from livekit.plugins import openai as openai_plugin

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

logger = logging.getLogger("nova-v201")
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

    async def refresh_instructions(self):
        """Rebuild instructions based on current state (called each turn)."""
        new_prompt = self.state.system_prompt()
        await self.update_instructions(new_prompt)

    async def on_user_turn_completed(self, chat_ctx, new_message):
        """Hook fired when kid finishes speaking. Refresh prompt + pace."""
        await self.refresh_instructions()
        await self.state.pace.acquire()


# ────────────────────────────────────────────────────────────────────────
# Per-room control channel — browser pushes game events via LiveKit data
# ────────────────────────────────────────────────────────────────────────
def register_data_handler(room: rtc.Room, state: NovaSessionState, session: AgentSession, agent: "NovaAgent"):
    """Listen for game events from the browser."""

    @room.on("data_received")
    def on_data(packet: rtc.DataPacket):
        try:
            raw = packet.data.decode("utf-8")
            # Raw-packet logger — proves a packet arrived at all, regardless of content
            logger.info(f"[PACKET] received {len(raw)} bytes: {raw[:120]}")
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
                    asyncio.create_task(_react_to_event(session, state, agent))

                # Phase transition to goodbye → speak final goodbye
                if event.get("event") == "phase" and event.get("phase") == "goodbye":
                    asyncio.create_task(_speak_goodbye(session, state, agent))

            elif kind == "vision-observation":
                obs = msg.get("text", "").strip()
                if obs and not state.vision_fired:
                    state.vision_fired = True
                    state.ctx.observed_visual = obs
                    logger.info(f"[data] vision observation: '{obs}'")
                    asyncio.create_task(_drop_in_observation(session, state, obs, agent))

            # ═══ TEST BENCH HANDLERS ═══
            elif kind == "test-utter":
                # Force Nova to say EXACTLY this text right now
                text = msg.get("text", "").strip()
                if text:
                    logger.info(f"[test] utter: '{text[:60]}'")
                    asyncio.create_task(_test_utter(session, state, text))

            elif kind == "test-inject":
                # Inject a persona overlay — Nova's next replies follow it
                overlay = msg.get("overlay", "").strip()
                trigger_speak = msg.get("trigger_speak", False)
                logger.info(f"[test] inject overlay: '{overlay[:80]}'")
                state.ctx.persona_overlay = overlay or None
                asyncio.create_task(agent.refresh_instructions())
                if trigger_speak:
                    asyncio.create_task(_test_speak_with_overlay(session, state, agent))

            elif kind == "test-clear-overlay":
                logger.info("[test] cleared persona overlay")
                state.ctx.persona_overlay = None
                asyncio.create_task(agent.refresh_instructions())

            elif kind == "test-force-phase":
                new_phase = msg.get("phase", "recognition")
                logger.info(f"[test] force phase → {new_phase}")
                state.push_event({"event": "phase", "phase": new_phase})

            elif kind == "user-said":
                # Kid typed instead of (or alongside) speaking. Treat as voice input.
                text = msg.get("text", "").strip()
                if text:
                    logger.info(f"[chat] user-said: '{text[:80]}'")
                    asyncio.create_task(_user_said(session, state, agent, text))

        except Exception as e:
            logger.error(f"[data] parse error: {e}")


# ────────────────────────────────────────────────────────────────────────
# Reaction helpers
# ────────────────────────────────────────────────────────────────────────
async def _react_to_event(session: AgentSession, state: NovaSessionState, agent: "NovaAgent"):
    """Game event happened — generate phase-aware reaction."""
    await state.pace.acquire()
    # Refresh prompt to current phase
    await agent.refresh_instructions()
    instructions = (
        f"React to game event '{state.ctx.last_event}'. "
        f"Current streak: {state.ctx.streak}. "
        f"Reply 1-6 words only. Follow your dance phase rules."
    )
    try:
        await session.generate_reply(instructions=instructions)
    except Exception as e:
        logger.error(f"[react] generate_reply failed: {e}")


async def _speak_goodbye(session: AgentSession, state: NovaSessionState, agent: "NovaAgent"):
    """Phase transitioned to goodbye — warm wrap-up."""
    await state.pace.acquire()
    memory.store.increment_sessions(state.kid_id)
    await agent.refresh_instructions()
    instructions = (
        f"The song ended. Speak warm goodbye now. "
        f"Stats: hits={state.ctx.hits}, max_streak={state.ctx.max_streak}. "
        f"Follow goodbye phase rules: ONE specific celebration + ONE open question."
    )
    try:
        await session.generate_reply(instructions=instructions)
    except Exception as e:
        logger.error(f"[goodbye] generate_reply failed: {e}")


async def _drop_in_observation(session: AgentSession, state: NovaSessionState, observation: str, agent: "NovaAgent"):
    """Drop in vision observation naturally."""
    await state.pace.acquire()
    await agent.refresh_instructions()
    instructions = (
        f"You just noticed: '{observation}'. "
        f"Say it warmly with '...' pauses, like you spotted it. ONE sentence."
    )
    try:
        await session.generate_reply(instructions=instructions)
    except Exception as e:
        logger.error(f"[vision] generate_reply failed: {e}")


# ────────────────────────────────────────────────────────────────────────
# TEST BENCH helpers — used by nova-test.html sandbox only
# ────────────────────────────────────────────────────────────────────────
async def _test_utter(session: AgentSession, state: NovaSessionState, text: str):
    """Force Nova to say EXACTLY this text. No LLM, no interpretation."""
    await state.pace.acquire()
    try:
        await session.say(text)
        logger.info(f"[test] uttered: '{text[:40]}'")
    except Exception as e:
        logger.error(f"[test] utter failed: {e}")


async def _test_speak_with_overlay(session: AgentSession, state: NovaSessionState, agent: "NovaAgent"):
    """After an overlay is injected, ask Nova to speak so we hear it apply."""
    await state.pace.acquire()
    await agent.refresh_instructions()
    try:
        await session.generate_reply(
            instructions="Speak ONE short sentence now, following any active override."
        )
    except Exception as e:
        logger.error(f"[test] overlay-speak failed: {e}")


async def _user_said(session: AgentSession, state: NovaSessionState, agent: "NovaAgent", text: str):
    """Browser sent text-as-voice. Inject as user input so Nova replies naturally."""
    logger.info(f"[TYPE] kid typed → '{text[:80]}'")
    await state.pace.acquire()
    await agent.refresh_instructions()
    logger.info("[BRAIN] generating reply to typed input...")
    try:
        await session.generate_reply(user_input=text)
        logger.info(f"[BRAIN] reply call returned for: '{text[:40]}'")
    except Exception as e:
        logger.exception(f"[BRAIN] generate_reply FAILED for typed input: {e}")


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
    logger.info(f"[nova-v201] entrypoint room={ctx.room.name}")

    kid_id = None
    try:
        if ctx.room.metadata:
            meta = json.loads(ctx.room.metadata)
            kid_id = meta.get("kidId")
    except Exception:
        pass

    state = NovaSessionState(kid_id=kid_id)
    logger.info(
        f"[nova-v201] kid_id={state.kid_id} "
        f"name={state.ctx.name} sessions_before={state.ctx.sessions_before}"
    )

    avatar_id = os.getenv("NOVA_AVATAR_ID", "e976bbb2-de60-4da6-845e-4b754050e55b")

    # ─────────────────────────────────────────────────────────────
    # Nova's brain: OpenAI GPT-4o-mini (single LLM).
    # WHY OpenAI not Gemini: Gemini blocked our fairy greeting as PROHIBITED
    # because "appeared + child + *gasp*" tripped its kids-safety classifier
    # (Jun 3 2026 logs). OpenAI handles kids' creative content normally.
    # WHY mini not full 4o: 10x cheaper, fast enough (<800ms), enough quality
    # for short fairy reactions. Switch via NOVA_OPENAI_MODEL env if needed.
    # ─────────────────────────────────────────────────────────────
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.error("[nova-v201] FATAL: OPENAI_API_KEY missing on worker")
        raise RuntimeError("OPENAI_API_KEY required — add it on Render → worker → Environment")

    llm_instance = openai_plugin.LLM(
        model=os.getenv("NOVA_OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("NOVA_TEMPERATURE", "0.85")),
        api_key=openai_key,
    )
    logger.info(f"[nova-v201] brain = OpenAI {os.getenv('NOVA_OPENAI_MODEL', 'gpt-4o-mini')}")

    # Build session pipeline
    session_kwargs = dict(
        # ─────────────────────────────────────────────────────────────
        # STT: OpenAI Whisper (gpt-4o-mini-transcribe).
        # Was: Deepgram. Confirmed broken on this account after 2 days of
        # logs showing `[MIC-IN] subscribed` but ZERO `[HEAR]` lines — meaning
        # audio reached Deepgram but no transcripts came back. Likely missing
        # account entitlement for multilingual streaming on nova-3.
        # Switched to OpenAI because the OPENAI_API_KEY is already on the
        # worker and proven working (the brain uses it). Same key, one vendor.
        # ─────────────────────────────────────────────────────────────
        stt=openai_plugin.STT(
            model=os.getenv("NOVA_STT_MODEL", "gpt-4o-mini-transcribe"),
            language=os.getenv("NOVA_STT_LANG", "en"),
            api_key=openai_key,
        ),
        llm=llm_instance,
        tts=elevenlabs.TTS(
            # FREYA: American young female (~20), bright/cheerful by default.
            # Lily was British — wrong accent for our product. Aria/Bella as fallbacks.
            voice_id=os.getenv("NOVA_VOICE_ID", "jsCqWAovK2LkecY7zXl4"),
            model=os.getenv("NOVA_TTS_MODEL", "eleven_flash_v2_5"),
            voice_settings=elevenlabs.VoiceSettings(
                # Tuned: less stable (more aliveness), higher similarity (less distortion).
                stability=float(os.getenv("NOVA_VOICE_STABILITY", "0.25")),
                similarity_boost=float(os.getenv("NOVA_VOICE_SIMILARITY", "0.85")),
                style=float(os.getenv("NOVA_VOICE_STYLE", "0.60")),
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
        logger.info("[nova-v201] turn detector enabled")

    # Watch for ANY remote track arriving at the worker — confirms mic plumbing
    @ctx.room.on("track_subscribed")
    def _on_remote_track(track, publication, participant):
        kind = getattr(track, "kind", "?")
        logger.info(f"[MIC-IN] subscribed to {kind} from {participant.identity}")

    @ctx.room.on("track_published")
    def _on_remote_pub(publication, participant):
        kind = getattr(publication, "kind", "?")
        logger.info(f"[MIC-IN] {participant.identity} published {kind} track")

    session = AgentSession(**session_kwargs)
    logger.info("[nova-v201] step 1: AgentSession created")

    # ─────────────────────────────────────────────────────────────
    # HEAVY LOGGING HOOKS — distinct log line at each pipeline stage.
    # Each step gets a tag so we can tell exactly WHERE a session breaks:
    #   [HEAR]  — Deepgram transcribed kid's voice
    #   [TYPE]  — kid typed (via user-said)
    #   [BRAIN] — Gemini being called / replied / failed
    #   [SPEAK] — Nova spoke (audio went out)
    #   [SILENT]— nothing came back from the brain
    # If you see [HEAR] but no [BRAIN], STT works but brain isn't picking up.
    # If [BRAIN] but no [SPEAK], brain replied but TTS/Runway failed.
    # ─────────────────────────────────────────────────────────────
    try:
        @session.on("user_input_transcribed")
        def _on_transcribed(ev):
            text = getattr(ev, "transcript", "") or getattr(ev, "text", "")
            is_final = getattr(ev, "is_final", True)
            if not text.strip():
                return
            # Log BOTH interim and final so we can see Deepgram is alive at all
            if is_final:
                logger.info(f"[HEAR] final ✓ kid voice → '{text[:80]}'")
            else:
                logger.info(f"[HEAR] interim … '{text[:60]}'")
    except Exception as e:
        logger.warning(f"[hook] user_input_transcribed unavailable: {e}")

    # Direct STT error visibility — if Deepgram is rejecting audio we'll see it
    try:
        @session.on("error")
        def _on_session_error(ev):
            err = getattr(ev, "error", None)
            logger.error(f"[STT-OR-AGENT-ERROR] {err}")
    except Exception:
        pass

    try:
        @session.on("agent_state_changed")
        def _on_agent_state(ev):
            new_state = getattr(ev, "new_state", None)
            old_state = getattr(ev, "old_state", None)
            if new_state:
                logger.info(f"[BRAIN] state {old_state} → {new_state}")
            speaking = (new_state == "speaking")
            state.pace.mark_speaking(speaking)
    except Exception as e:
        logger.warning(f"[hook] agent_state_changed unavailable: {e}")

    try:
        @session.on("conversation_item_added")
        def _on_item(ev):
            item = getattr(ev, "item", None)
            if not item: return
            role = getattr(item, "role", "?")
            txt = getattr(item, "text_content", None) or ""
            if isinstance(txt, list): txt = " ".join(str(x) for x in txt)
            txt = str(txt)[:140]
            if role == "assistant" and txt:
                logger.info(f"[SPEAK] Nova said → '{txt}'")
            elif role == "user" and txt:
                logger.info(f"[HEAR] confirmed user msg → '{txt}'")
    except Exception as e:
        logger.warning(f"[hook] conversation_item_added unavailable: {e}")

    # Runway face plugin
    try:
        runway_avatar = runway.AvatarSession(avatar_id=avatar_id)
        await runway_avatar.start(session, room=ctx.room)
        logger.info(f"[nova-v201] step 2: runway avatar started, id={avatar_id[:8]}")
    except Exception as e:
        logger.exception(f"[nova-v201] CRASH at runway start: {e}")
        raise

    # The agent
    agent = NovaAgent(state)
    logger.info("[nova-v201] step 3: NovaAgent created")

    # Data channel listener BEFORE session starts (catch early events)
    register_data_handler(ctx.room, state, session, agent)
    logger.info("[nova-v201] step 4: data handler registered")

    # Import RoomInputOptions for explicit subscribe config
    from livekit.agents.voice.room_io import RoomInputOptions

    try:
        await session.start(
            agent=agent,
            room=ctx.room,
            # CRITICAL: explicitly subscribe to standard participants (kids) for
            # both audio AND text. Without this, in v1.5.16 the agent sometimes
            # links to the wrong participant (the runway-avatar agent that joins
            # the room) and never hears the kid. This was the real bug behind
            # "Nova never replies to anything I say or type."
            room_input_options=RoomInputOptions(
                audio_enabled=True,
                text_enabled=True,
                video_enabled=False,
                participant_kinds=[rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD],
            ),
        )
        logger.info("[nova-v201] step 5: session.start COMPLETE (kid-audio + text subscribed)")
    except Exception as e:
        logger.exception(f"[nova-v201] CRASH at session.start: {e}")
        raise

    logger.info("[nova-v201] step 6: pipeline ready, heavy logging active")

    # GREETING — first words from OUR brain
    state.greeting_done = True
    logger.info("[nova-v201] step 7: about to generate greeting...")

    if state.ctx.name and state.ctx.sessions_before > 0:
        greet_instructions = (
            f"{state.ctx.name} just came back for another session. "
            f"Greet warmly, like a friend. ONE short sentence using their name once."
        )
        fallback_greeting = f"hey {state.ctx.name}! you came back."
    else:
        greet_instructions = (
            "You just appeared. Greet the kid warmly, say your name is Nova, "
            "ask their name. ONE short flowing sentence."
        )
        fallback_greeting = "hey! I'm Nova... what's your name?"

    try:
        # 10s timeout — if generate_reply hangs (LLM timeout), don't kill the session
        await asyncio.wait_for(
            session.generate_reply(instructions=greet_instructions),
            # First OpenAI call from a fresh worker takes ~5-12s (TLS handshake
            # + model warmup). 10s wasn't enough; we saw the fallback fire even
            # on healthy sessions. 20s is comfortable without making the user wait
            # forever if something is truly wrong.
            timeout=20.0,
        )
        logger.info("[nova-v201] step 8: GREETING SENT SUCCESSFULLY (via LLM)")
    except asyncio.TimeoutError:
        logger.warning("[nova-v201] greeting LLM timed out → falling back to plain say()")
        try:
            await session.say(fallback_greeting)
            logger.info("[nova-v201] step 8: GREETING SENT (fallback)")
        except Exception as e2:
            logger.error(f"[nova-v201] fallback greeting also failed: {e2}")
    except Exception as e:
        logger.exception(f"[nova-v201] greeting failed (non-timeout): {e}")
        try:
            await session.say(fallback_greeting)
            logger.info("[nova-v201] step 8: GREETING SENT (fallback after error)")
        except Exception as e2:
            logger.error(f"[nova-v201] fallback greeting also failed: {e2}")

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
