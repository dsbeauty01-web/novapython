"""
Nova v207 — LiveKit Agent worker (Loora-grade brain architecture)

Brain = OpenAI gpt-4o-mini with FIVE-LAYER context prompts:
  L1 IDENTITY        — locked persona (~500 tokens)
  L2 KID PROFILE     — from memory.py (Postgres or RAM)
  L3 KID KNOWLEDGE   — from knowledge.py (colors/animals/foods/etc.)
  L4 SESSION STATE   — phase, music sec, recent events, message history
  L5 PHASE PERSONA   — recognition / dance / goodbye

Reaction tiering (router in personality.reaction_tier):
  Tier 1  phrase_bank — 80% of dance reactions, free, ~50ms
  Tier 2  llm_micro   — milestone streaks (3,5,10), first_hit, ~500ms
  Tier 3  llm_rich    — kid speech, goodbye, vision, ~700ms

Pipeline:
    Kid voice (browser Web Speech API) ──► data channel ──► worker
                                                                │
                                                                ▼
    OpenAI gpt-4o-mini ◄── 5-layer prompt ◄── personality.build_system_prompt()
                                                                │
                                                                ▼
    text ──► ElevenLabs Freya TTS ──► Runway face lipsync ──► kid

Memory (memory.py):
  - Postgres if DATABASE_URL set on Render (persistent)
  - RAM fallback otherwise (per-process)
  - Per kid: name, sessions, streaks, shared_facts (pets/colors/etc.),
    best_moments, message_history (last 12 turns), energy_read

Character (personality.py):
  Warm American ~20yo dance friend. Cool-older-cousin energy.
  BANNED: "amazing", "awesome", "great job", baby-talk, fairy-isms.
  Mandated: specific praise, mirror-and-echo, gentle correction.

History of brain choices:
- v113: Runway's hidden brain (no control, abandoned)
- v200: Anthropic Haiku (kept timing out)
- v201: Gemini Flash (blocked content as PROHIBITED)
- v202-206: OpenAI gpt-4o-mini (industry standard) + flat prompts
- v207 now: OpenAI gpt-4o-mini + 5-layer prompts + tier router + knowledge base

Heavy logging: [HEAR]/[TYPE]/[BRAIN]/[SPEAK]/[PACKET]/[MIC-IN]/[react/...] tags.
"""

# ══════════════════════════════════════════════════════════════════════
# v225 — VOICE LATENCY TUNING (live-entertainment feel)
#
# Optimized for ~400ms response: Silero VAD trimmed (0.4s silence), VAD
# turn-detection (not semantic), interruptions on, ElevenLabs Flash v2.5
# streaming TTS (fast first byte), and short dance-phase replies (max 80
# tokens). The avatar lipsync may lag the audio slightly — that drift is
# ACCEPTABLE for fast-paced kid interaction, where snappy back-and-forth
# beats perfect mouth-sync. Warmth still lands via personality.py phrasing
# and the voice settings (stability 0.20 / style 0.85), not the TTS model.
# ══════════════════════════════════════════════════════════════════════

import os
import json
import time
import wave
import random
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

logger = logging.getLogger("nova-v207")
logging.basicConfig(level=logging.INFO)


# ──────────────────────────────────────────────────────────────────────
# PRE-CACHED FILLER SYSTEM — perceived-magic latency win.
#
# On STT-final we instantly play ONE tiny pre-generated clip ("ooh!", "yes!",
# "lightning!"…) through Nova's avatar while the real LLM reply is produced; the
# real reply then plays right after, so the kid feels an instant reaction.
#
# Clips: audio/fillers/*.wav (24kHz mono PCM) — decoded with the stdlib `wave`
# module so NO extra dependency / build step is needed. FULLY FAIL-SAFE: any
# error loads/plays disables fillers and never touches the real-reply path.
# ──────────────────────────────────────────────────────────────────────
FILLER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio", "fillers")
FILLER_NAMES = ["ooh", "yes", "wait", "woohoo", "hmm", "boom", "almost", "lightning"]


class FillerPlayer:
    GAP_SEC = 2.0          # min seconds between fillers (no chaos)
    MIN_TEXT_CHARS = 3     # skip very short utterances

    _clips = None          # class-level cache: name -> (pcm_bytes, rate, channels)

    @classmethod
    def _load_clips(cls):
        if cls._clips is not None:
            return
        cls._clips = {}
        for n in FILLER_NAMES:
            p = os.path.join(FILLER_DIR, n + ".wav")
            try:
                if os.path.exists(p):
                    with wave.open(p, "rb") as w:
                        cls._clips[n] = (
                            w.readframes(w.getnframes()),
                            w.getframerate(),
                            w.getnchannels(),
                        )
            except Exception as e:
                logger.error(f"[filler] failed to load {n}.wav: {e}")
        logger.info(f"[filler] loaded {len(cls._clips)} clips from {FILLER_DIR}")

    def __init__(self):
        try:
            self._load_clips()
        except Exception as e:
            logger.error(f"[filler] _load_clips crashed, disabling: {e}")
        self.last_name = None
        self.last_fire = 0.0
        self.enabled = bool(type(self)._clips)

    def _pick(self):
        names = [n for n in self._clips if n != self.last_name] or list(self._clips)
        return random.choice(names)

    async def _frames(self, pcm, rate, ch):
        spc = max(1, rate // 100)      # 10ms worth of samples
        cb = spc * ch * 2              # bytes per 10ms frame (16-bit)
        for i in range(0, len(pcm), cb):
            chunk = pcm[i:i + cb]
            if len(chunk) < cb:
                chunk = chunk + b"\x00" * (cb - len(chunk))
            yield rtc.AudioFrame(
                data=chunk, sample_rate=rate, num_channels=ch,
                samples_per_channel=len(chunk) // (ch * 2),
            )

    def should_fire(self, text, is_speaking):
        if not self.enabled:
            return False
        if is_speaking:                                       # Nova already talking
            return False
        if text is not None and len(text.strip()) < self.MIN_TEXT_CHARS:
            return False
        if time.time() - self.last_fire < self.GAP_SEC:       # too soon
            return False
        return True

    async def fire(self, session, text):
        """Play one filler clip. Fully isolated — never raises into the turn."""
        try:
            name = self._pick()
            self.last_name = name
            self.last_fire = time.time()
            logger.info(f"[filler] CHOSE '{name}' t={self.last_fire:.3f} user='{(text or '')[:30]}'")
            pcm, rate, ch = self._clips[name]
            logger.info(f"[filler] PLAYING '{name}' (real LLM reply being produced in parallel)")
            await session.say(
                name, audio=self._frames(pcm, rate, ch),
                add_to_chat_ctx=False, allow_interruptions=False,
            )
            logger.info(f"[filler] DONE '{name}'")
        except Exception as e:
            logger.error(f"[filler] play failed -> disabling fillers for safety: {e}")
            self.enabled = False


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

    # Backup floor, in seconds. Was 1.2s but kid complained it felt slow.
    # 0.3s = barely perceptible gap; Nova's natural speaking-state event
    # already gates her from talking over herself.
    MIN_GAP_SEC = float(os.getenv("NOVA_MIN_GAP_SEC", "0.3"))

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

    def bump(self, k):
        """Safely increment a per-session metric counter (never raises)."""
        try:
            self.metrics[k] = self.metrics.get(k, 0) + 1
        except Exception:
            pass

    def __init__(self, kid_id: Optional[str] = None):
        self.kid_id = kid_id or f"anon-{int(time.time())}"
        self.ctx = personality.NovaContext(phase="recognition")
        self.pace = PaceGate()
        self.vision_fired = False
        self.greeting_done = False
        self.session_started_at = time.time()
        # Per-session telemetry counters → logged as [SESSION-SUMMARY] at teardown
        self.t_start = time.time()
        self.metrics = {"turns": 0, "replies": 0, "fillers": 0, "errors": 0}
        # Load full kid profile from memory (Postgres or RAM)
        mem = memory.store.get(self.kid_id)
        # Always copy what we have — even partial profile helps Nova
        self.ctx.name = mem.name
        self.ctx.sessions_before = mem.total_sessions
        self.ctx.max_streak = mem.max_streak
        self.ctx.favorite_move = mem.favorite_move
        self.ctx.favorite_song = mem.favorite_song
        self.ctx.shared_facts = dict(mem.shared_facts) if mem.shared_facts else {}
        self.ctx.best_moments_history = list(mem.best_moments) if mem.best_moments else []
        self.ctx.energy_read = mem.energy_read or "unknown"
        self.ctx.message_history = list(mem.message_history) if mem.message_history else []
        if mem.best_moments:
            self.ctx.best_moment = mem.best_moments[-1]
        logger.info(f"[memory] loaded kid={self.kid_id} name={mem.name} "
                    f"sessions={mem.total_sessions} max_streak={mem.max_streak} "
                    f"facts={len(mem.shared_facts)}")

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
        """Hook fired when kid finishes speaking (STT final). Fire an instant
        pre-cached filler while the real reply is produced, then refresh + pace."""
        try:
            fp = getattr(self.state, "filler", None)
            sess = getattr(self.state, "session", None)
            if fp and sess:
                txt = None
                try:
                    tc = getattr(new_message, "text_content", None)
                    txt = tc() if callable(tc) else tc
                except Exception:
                    txt = None
                if fp.should_fire(txt, self.state.pace._is_speaking):
                    self.state.bump("fillers")
                    asyncio.create_task(fp.fire(sess, txt))
        except Exception as e:
            logger.error(f"[filler] turn hook error: {e}")
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

                # Phase transition to dance → fire hype intro line ("ready?")
                # Worker generates ONE short line. This replaces the old
                # browser-side fake "user-said" stage direction that leaked.
                if event.get("event") == "phase" and event.get("phase") == "dance":
                    asyncio.create_task(_speak_dance_intro(session, state, agent))

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

            elif kind == "client-log":
                # Browser telemetry batch — print each entry so the full session
                # (client + server) lands in ONE log stream, correlated by time.
                try:
                    for ev in (msg.get("events") or [])[:60]:
                        t = ev.get("t"); tag = ev.get("tag"); m = str(ev.get("msg", ""))[:180]
                        d = ev.get("data")
                        extra = ""
                        if d:
                            try:
                                extra = " " + json.dumps(d)[:220]
                            except Exception:
                                extra = ""
                        logger.info(f"[CLIENT] t={t} [{tag}] {m}{extra}")
                except Exception as e:
                    logger.error(f"[client-log] handler error: {e}")

        except Exception as e:
            logger.error(f"[data] parse error: {e}")


# ────────────────────────────────────────────────────────────────────────
# Reaction helpers
# ────────────────────────────────────────────────────────────────────────
async def _react_to_event(session: AgentSession, state: NovaSessionState, agent: "NovaAgent"):
    """Game event happened — pick tier (phrase bank vs LLM) for speed + cost."""
    event_name = state.ctx.last_event or "hit"
    tier = personality.reaction_tier(event_name, state.ctx.streak)

    # Drop reaction entirely if Nova is still mid-speech — don't queue overlap.
    # (Game cues fire fast; we'd rather skip than stack.)
    if state.pace._is_speaking:
        logger.info(f"[react] SKIP {event_name} (Nova still speaking)")
        return

    await state.pace.acquire()
    await agent.refresh_instructions()

    if tier == "phrase_bank":
        # Tier 1: instant, free, from PHRASE_BANKS
        line = personality.pick_phrase(event_name, state.ctx.streak, state.ctx.name)
        if not line:
            return
        try:
            await session.say(line)
            logger.info(f"[react/bank] {event_name} streak={state.ctx.streak} → '{line}'")
        except Exception as e:
            logger.error(f"[react/bank] say failed: {e}")
        return

    # Tier 2: short LLM call for milestones (streak 3, 5, 10) or first_hit
    instructions = (
        f"React to game event '{event_name}' with streak {state.ctx.streak}. "
        f"1-6 WORDS ONLY. Follow dance-phase rules."
    )
    try:
        await session.generate_reply(instructions=instructions)
        logger.info(f"[react/llm] {event_name} streak={state.ctx.streak}")
    except Exception as e:
        logger.error(f"[react/llm] generate_reply failed: {e}")


async def _speak_dance_intro(session: AgentSession, state: NovaSessionState, agent: "NovaAgent"):
    """Phase transitioned to dance — say ONE hype line over the countdown."""
    await state.pace.acquire()
    await agent.refresh_instructions()
    instructions = (
        "Kid just hit dance. Say ONE short hype line — 1 to 3 words only "
        "(like 'ready?', 'okay let's go!', 'here we go!'). No questions. "
        "Just a quick cheer."
    )
    try:
        await session.generate_reply(instructions=instructions)
    except Exception as e:
        logger.error(f"[dance-intro] generate_reply failed: {e}")


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
    """Kid spoke or typed. Inject as user input + extract knowledge + save to memory."""
    logger.info(f"[TYPE] kid typed → '{text[:80]}'")

    # Update live context so next prompt build has the kid's words +
    # knowledge.py can detect topic mentions (colors, animals, foods)
    state.ctx.last_kid_text = text

    # Persistent message history — survives across sessions if Postgres on
    try:
        memory.store.add_message(state.kid_id, "user", text)
    except Exception as e:
        logger.warning(f"[memory] add_message failed: {e}")

    # Naive shared-fact harvest: catch common "my X is Y" patterns
    # so Nova remembers "I have a cat named Mango" next session
    _harvest_facts(state, text)

    # FILLER: instant reaction while the real reply is produced (typed path)
    try:
        fp = getattr(state, "filler", None)
        if fp and fp.should_fire(text, state.pace._is_speaking):
            state.bump("fillers")
            asyncio.create_task(fp.fire(session, text))
    except Exception as e:
        logger.error(f"[filler] user-said hook error: {e}")
    await state.pace.acquire()
    await agent.refresh_instructions()
    logger.info("[BRAIN] generating reply to kid input...")
    try:
        await session.generate_reply(user_input=text)
        logger.info(f"[BRAIN] reply call returned for: '{text[:40]}'")
    except Exception as e:
        logger.exception(f"[BRAIN] generate_reply FAILED for kid input: {e}")


def _harvest_facts(state: NovaSessionState, text: str):
    """Very lightweight pattern-match for facts kid shares.
    Catches: 'my [thing] is [name]', 'I have a [pet]', 'I like [thing]'.
    Real NLU would be a Tier-3 LLM call; this is the cheap version."""
    import re
    t = text.lower().strip()
    patterns = [
        (r"my\s+(?:name|cat|dog|pet|brother|sister|friend|mom|dad)\s+is\s+([a-z][a-z\-' ]{1,30})",
         lambda m: ("relation_subject", m.group(0).strip())),
        (r"i\s+(?:have|got)\s+a\s+([a-z][a-z\-' ]{1,30})", lambda m: ("has", m.group(1).strip())),
        (r"i\s+(?:like|love)\s+([a-z][a-z\-' ]{1,30})", lambda m: ("likes", m.group(1).strip())),
        (r"my\s+favorite\s+(?:color|food|song|move|animal)\s+is\s+([a-z][a-z\-' ]{1,30})",
         lambda m: ("favorite", m.group(0).strip())),
    ]
    found = []
    for pat, fn in patterns:
        m = re.search(pat, t)
        if m:
            key, val = fn(m)
            val = val.strip().rstrip(".!?,")
            if 1 < len(val) < 40:
                found.append((key, val))
    if not found:
        return
    for key, val in found[:2]:
        state.ctx.shared_facts[key] = val
        try:
            memory.store.add_shared_fact(state.kid_id, key, val)
            logger.info(f"[fact] {state.kid_id}: {key}={val}")
        except Exception as e:
            logger.warning(f"[fact] save failed: {e}")


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
    logger.info(f"[nova-v207] entrypoint room={ctx.room.name}")

    kid_id = None
    try:
        if ctx.room.metadata:
            meta = json.loads(ctx.room.metadata)
            kid_id = meta.get("kidId")
    except Exception:
        pass

    state = NovaSessionState(kid_id=kid_id)
    logger.info(
        f"[nova-v207] kid_id={state.kid_id} "
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
        logger.error("[nova-v207] FATAL: OPENAI_API_KEY missing on worker")
        raise RuntimeError("OPENAI_API_KEY required — add it on Render → worker → Environment")

    llm_instance = openai_plugin.LLM(
        model=os.getenv("NOVA_OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("NOVA_TEMPERATURE", "0.85")),
        api_key=openai_key,
        # v225-fix: REMOVED `max_completion_tokens=...` here. That kwarg is not
        # accepted by this pinned livekit-plugins-openai version, so constructing
        # the LLM raised TypeError on worker boot → the "nova" agent never joined
        # the room → Nova "did not connect" on every frontend. Replies are short
        # by design anyway, so dropping the hard cap costs little. (If you want a
        # token cap later, set it per-request, not on the constructor.)
    )
    logger.info(f"[nova-v207] brain = OpenAI {os.getenv('NOVA_OPENAI_MODEL', 'gpt-4o-mini')}")

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
            # CRITICAL: enable realtime streaming. Without this, STT only
            # transcribes ONCE per turn after VAD says the kid stopped talking
            # — and if VAD never fires, no transcript ever appears (silent fail).
            # Realtime mode streams interim + final transcripts continuously.
            use_realtime=True,
        ),
        llm=llm_instance,
        tts=elevenlabs.TTS(
            # MATILDA — warm female, kid-friendly (rebuild sprint, replaces Freya).
            # model eleven_flash_v2_5: Nova is English-only; Flash ~75ms TTFA vs
            #   500-800ms for multilingual_v2.
            # settings: stability 0.50 (steadier; was 0.20 erratic),
            #   style 0.40 (less theatrical; was 0.85).
            voice_id=os.getenv("NOVA_VOICE_ID", "XrExE9yKIg1WjnnlVkGX"),
            model=os.getenv("NOVA_TTS_MODEL", "eleven_flash_v2_5"),
            voice_settings=elevenlabs.VoiceSettings(
                stability=float(os.getenv("NOVA_VOICE_STABILITY", "0.50")),
                similarity_boost=float(os.getenv("NOVA_VOICE_SIMILARITY", "0.85")),
                style=float(os.getenv("NOVA_VOICE_STYLE", "0.40")),
                use_speaker_boost=True,
            ),
        ),
        vad=silero.VAD.load(
            # v225: trimmed for snappy turn-taking. 0.4s of silence ends a turn
            # (was the ~0.55 default), 0.05s min speech, 0.2s prefix padding.
            min_speech_duration=0.05,
            min_silence_duration=0.4,
            prefix_padding_duration=0.2,
            activation_threshold=0.5,
        ),
        # v225: VAD-based turn detection (no semantic model file needed), let the
        # kid barge in over Nova, and count just 0.4s of speech as an interruption.
        turn_detection="vad",
        allow_interruptions=True,
        min_interruption_duration=0.4,
    )
    # Turn-detector disabled: its model file (model_q8.onnx) wasn't available
    # in this environment. Silero VAD (already configured) handles turn-taking
    # just fine. Re-enable later if you choose to bake the model into the build.
    if False:  # was: if TURN_DETECTOR_AVAILABLE:
        session_kwargs["turn_detection"] = MultilingualModel()
        logger.info("[nova-v207] turn detector enabled")

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
    logger.info("[nova-v207] step 1: AgentSession created")

    # Wire the pre-cached filler system (instant reactions while the LLM cooks).
    # Guarded so a filler problem can never block the session from starting.
    state.session = session
    try:
        state.filler = FillerPlayer()
        logger.info(f"[filler] system ready (enabled={state.filler.enabled})")
    except Exception as e:
        logger.error(f"[filler] init failed, fillers disabled: {e}")
        state.filler = None

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
                state.bump("turns")
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
            state.bump("errors")
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
                state.bump("replies")
                logger.info(f"[SPEAK] Nova said → '{txt}'")
                # Save Nova's reply to history — multi-turn memory survives
                try:
                    memory.store.add_message(state.kid_id, "assistant", txt)
                except Exception as e:
                    logger.warning(f"[memory] add_message(assistant) failed: {e}")
            elif role == "user" and txt:
                logger.info(f"[HEAR] confirmed user msg → '{txt}'")
    except Exception as e:
        logger.warning(f"[hook] conversation_item_added unavailable: {e}")

    # Runway face plugin
    try:
        runway_avatar = runway.AvatarSession(avatar_id=avatar_id)
        await runway_avatar.start(session, room=ctx.room)
        logger.info(f"[nova-v207] step 2: runway avatar started, id={avatar_id[:8]}")
    except Exception as e:
        logger.exception(f"[nova-v207] CRASH at runway start: {e}")
        raise

    # The agent
    agent = NovaAgent(state)
    logger.info("[nova-v207] step 3: NovaAgent created")

    # Data channel listener BEFORE session starts (catch early events)
    register_data_handler(ctx.room, state, session, agent)
    logger.info("[nova-v207] step 4: data handler registered")

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
                # CRITICAL: default sample rate is 24000Hz but Whisper expects
                # 16000Hz. Wrong rate = no transcripts (silent fail). This may
                # also be why Deepgram silently produced nothing.
                audio_sample_rate=16000,
            ),
        )
        logger.info("[nova-v207] step 5: session.start COMPLETE (kid-audio + text subscribed)")
    except Exception as e:
        logger.exception(f"[nova-v207] CRASH at session.start: {e}")
        raise

    logger.info("[nova-v207] step 6: pipeline ready, heavy logging active")

    # GREETING — first words from OUR brain
    state.greeting_done = True
    logger.info("[nova-v207] step 7: about to generate greeting...")

    # Rebuild sprint: HARDCODED first greeting — no LLM wait at session start
    # (saves ~2-3s, consistent first impression). The LLM drives every reply
    # AFTER this opening line.
    if state.ctx.name and state.ctx.sessions_before > 0:
        first_line = f"Ohh — {state.ctx.name}! You're back!"
    else:
        first_line = "Ohh hi! I'm Nova! What's your name?"

    try:
        await session.say(first_line)
        logger.info(f"[nova-v207] step 8: GREETING SENT (hardcoded): '{first_line}'")
    except Exception as e:
        logger.error(f"[nova-v207] hardcoded greeting failed: {e}")

    # Kick off vision request in background
    asyncio.create_task(_vision_trigger_loop(ctx.room, state))

    # Gentle idle engagement — soft presence if the child goes quiet
    asyncio.create_task(_idle_watch_loop(session, state))

    # Per-session AUTO-SUMMARY: logged once when the session tears down, so EVERY
    # session self-reports its headline metrics into the log stream (option A).
    async def _log_session_summary():
        try:
            dur = round(time.time() - state.t_start, 1)
            logger.info("[SESSION-SUMMARY] " + json.dumps({
                "kid": state.kid_id, "dur_s": dur, "phase": state.ctx.phase, **state.metrics
            }))
        except Exception as e:
            logger.error(f"[summary] failed: {e}")
    try:
        ctx.add_shutdown_callback(_log_session_summary)
    except Exception as e:
        logger.warning(f"[summary] could not register shutdown callback: {e}")


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
