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
# Authentic loora-voice thinking-sounds (re-generated Jun 18 2026). Short (0.5-0.7s)
# empathic/surprised interjections that play while she thinks — they cover latency
# AND warm her FACE (Runway lip-syncs whatever audio plays). Varied so she never
# repeats back-to-back (_pick avoids the last one).
# NEUTRAL, context-free listening sounds only (~0.4-0.6s). Meaning-laden words
# ("wait/whoa/yay/okay") are OUT — they imply a reaction that often contradicts
# the real reply ("doesn't make sense"). These are pure "I'm here, thinking" sounds
# that fit ANY reply and also warm her face via lip-sync.
# NOTE: "ahh" removed — it's a long open-vowel ("ahhhh") that sounds strangled/
# trembly ("like Parkinson") when the real reply interrupts it mid-vowel. Closed
# sounds (mm/mmhm/hmm) and short rounded ones (ooh/ohh) cut cleanly; open held
# vowels don't. This list matches the safe backchannel set (mm/mmhm/ooh/ohh/hmm).
FILLER_NAMES = ["mm", "mmhm", "ooh", "ohh", "hmm"]


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
        # OFF by default. Every clip goes through Runway lipsync (~500ms tax), so a
        # pre-reply "thinking sound" can't reliably land BEFORE the real reply — it
        # arrives late and overlaps/follows her sentence ("...move game? AHHH"),
        # which sounds nonsensical. The avatar pipeline makes the tiny-instant-filler
        # pattern structurally impossible. Re-enable for experiments with NOVA_FILLERS=1.
        self.enabled = bool(type(self)._clips) and os.getenv("NOVA_FILLERS", "0") == "1"
        logger.info(f"[filler] enabled={self.enabled} (NOVA_FILLERS={os.getenv('NOVA_FILLERS', '0')})")

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

    def claim(self, text, is_speaking):
        """Atomically decide + reserve a filler, SYNCHRONOUSLY. Returns the chosen
        clip name (to hand to fire) or None.

        Why synchronous: the v225 browser delivers each kid utterance TWICE — the
        published mic (-> STT -> on_user_turn_completed) AND a 'user-said' data
        packet (-> _user_said). Both hooks used to call should_fire()+fire(), and
        because last_fire was only stamped *inside* the async fire() task (which
        runs later), BOTH passed the gap check before either stamped -> two clips
        played = the 'ooo ooo' double-voice. Stamping last_fire here, on the event
        loop, before create_task, means the second hook sees the gap and skips."""
        if not self.should_fire(text, is_speaking):
            return None
        name = self._pick()
        self.last_name = name
        self.last_fire = time.time()      # reserve NOW so the twin hook can't double-fire
        logger.info(f"[filler] CLAIM '{name}' t={self.last_fire:.3f} user='{(text or '')[:30]}'")
        return name

    async def fire(self, session, name):
        """Play one already-claimed filler clip. Fully isolated — never raises
        into the turn. INTERRUPTIBLE: the real reply (produced in parallel) cuts
        the clip the instant it's ready, so the filler only covers the think-gap
        instead of *adding* its full length in front of every reply."""
        try:
            pcm, rate, ch = self._clips[name]
            logger.info(f"[filler] PLAYING '{name}' (real reply being produced in parallel)")
            await session.say(
                name, audio=self._frames(pcm, rate, ch),
                add_to_chat_ctx=False, allow_interruptions=True,
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
def _voice_settings():
    """Build ElevenLabs VoiceSettings, including `speed` if the installed plugin
    supports it. Older plugin versions lack the field — omit rather than crash."""
    base = dict(
        stability=float(os.getenv("NOVA_VOICE_STABILITY", "0.75")),
        similarity_boost=float(os.getenv("NOVA_VOICE_SIMILARITY", "0.90")),
        style=float(os.getenv("NOVA_VOICE_STYLE", "0.30")),
        use_speaker_boost=True,
    )
    # Calmer, unhurried delivery (was 0.92). Lower = slower/calmer.
    speed = float(os.getenv("NOVA_VOICE_SPEED", "0.88"))
    try:
        vs = elevenlabs.VoiceSettings(**base, speed=speed)
        logger.info(f"[nova-v207] VoiceSettings: speed={speed} SUPPORTED by plugin")
        return vs
    except TypeError:
        logger.warning("[nova-v207] VoiceSettings has NO 'speed' field — omitted (plugin too old)")
        return elevenlabs.VoiceSettings(**base)


def _inject_pauses(s):
    """Insert '...' micro-pauses after , ! ? so Nova sounds thoughtful, not blurted.
    Pure string ops (no regex import); safe on streamed chunks."""
    if not s:
        return s
    for p in ("! ", "? ", ", "):
        s = s.replace(p, p[0] + "... ")
    return s


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
        self.active = True  # flipped False at teardown to stop the idle loop ghost
        self.client_ready = asyncio.Event()  # set when browser finishes its reveal beat
        # ── Move-play game state ──
        self.game_started = False
        self.moves_done = 0
        self.game_done = asyncio.Event()        # kid signalled done OR 5-min cap
        self.kid_spoke = asyncio.Event()        # pulsed whenever the kid speaks
        self.last_kid_signal = None             # 'yes' | 'done' | None
        self.last_kid_text = None
        self.last_kid_speech_at = 0.0
        self.vision_waiter = None               # asyncio.Future per vision request
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
            # P3: clean isolation (only the cued part moved) → special praise bank
            self.ctx.last_event = "clean_hit" if event.get("clean") else "hit"
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

        elif ev == "age":
            # frontend's indirect age read → adapt Nova's tone (LITTLE/KID/TEEN/ADULT)
            tier = (event.get("tier") or "").strip().upper()
            if tier in ("LITTLE", "KID", "TEEN", "ADULT"):
                self.ctx.age_tier = tier
                logger.info(f"[state] age_tier = {tier}")

        elif ev == "energy":
            # energy mirror — frontend reports the kid's movement energy; Nova matches it
            lvl = (event.get("level") or "").strip().lower()
            if lvl in ("low", "med", "high"):
                self.ctx.energy_read = lvl

        elif ev == "move_cue":
            # nova-join / nova-wave: a new move card just opened. Remember which body
            # part is cued so hit reactions can NAME it ("nice head!"). No speech here
            # (the cue fires ~every beat — naming every one would be chatter).
            action = (event.get("action") or "").strip()
            self.ctx.current_move = personality.move_friendly(action) if action else None

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

    async def tts_node(self, text, model_settings):
        """Inject '...' micro-pauses into EVERY spoken line before TTS, so Nova
        sounds thoughtful. Falls back to the raw text stream on any error so it
        can never break her voice."""
        async def _paused():
            async for chunk in text:
                try:
                    yield _inject_pauses(chunk)
                except Exception:
                    yield chunk
        source = text
        try:
            source = _paused()
        except Exception:
            source = text
        async for frame in Agent.default.tts_node(self, source, model_settings):
            yield frame

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
                name = fp.claim(txt, self.state.pace._is_speaking)
                if name:
                    self.state.bump("fillers")
                    asyncio.create_task(fp.fire(sess, name))
        except Exception as e:
            logger.error(f"[filler] turn hook error: {e}")
        # LATENCY: do NOT call refresh_instructions() here — mutating the chat
        # context inside on_user_turn_completed cancels LiveKit's preemptive
        # (speculative) generation, adding ~200-400ms to every reply. The prompt
        # is kept fresh by phase changes + the ambient-vision loop instead.
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
                if obs:
                    state.ctx.observed_visual = obs
                    logger.info(f"[data] vision observation: '{obs}'")
                    # Move-game path: resolve whoever is awaiting this frame.
                    w = state.vision_waiter
                    if w is not None and not w.done():
                        w.set_result(obs)
                    # Legacy one-shot drop-in (only when NOT mid-game).
                    elif not state.game_started and not state.vision_fired:
                        state.vision_fired = True
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

            elif kind == "client-ready":
                # Browser finished its reveal beat — NOW Nova may greet (cinematic
                # reveal-then-greet). Greeting otherwise fires on an 8s fallback.
                logger.info("[ready] client-ready received → releasing greeting")
                state.client_ready.set()

        except Exception as e:
            logger.error(f"[data] parse error: {e}")


# ────────────────────────────────────────────────────────────────────────
# Reaction helpers
# ────────────────────────────────────────────────────────────────────────
def _evi_on() -> bool:
    # TEMP (2026-07-01): EVI forced OFF so Nova greets via the proven
    # Deepgram→LLM→ElevenLabs path (session.say). The Hume EVI websocket greeting
    # was timing out silently → Nova mute. To bring EVI back later WITHOUT a code
    # change, set env NOVA_FORCE_ELEVENLABS=0 (and USE_EVI=1); or delete this block.
    if os.getenv("NOVA_FORCE_ELEVENLABS", "1") == "1":
        return False
    return os.getenv("USE_EVI", "").lower() in ("1", "true", "yes", "on")


async def _nova_say(session: AgentSession, line: str):
    """Speak an EXACT line, EVI-safe. THE one speech path.
    Under EVI, session.say() is unsupported AND generate_reply(instructions=) speaks the
    given text verbatim (assistant_input) — so we pass the FINAL LINE (never a meta-prompt).
    Without EVI, session.say() speaks the line directly. Either way: Nova says the LINE,
    never the prompt. (This is the fix for 'Nova read the AI instructions out loud'.)"""
    if not line:
        return
    try:
        if _evi_on():
            await session.generate_reply(instructions=line)
        else:
            await session.say(line)
        logger.info(f"[SAY] '{line[:60]}'")
    except Exception as e:
        logger.error(f"[SAY] failed: {e}")


_OAI_CLIENT = None
async def _llm_line(system: str, user: str, max_tokens: int = 40) -> str:
    """Generate ONE short line via a cheap LLM (for EVI, which can't generate-from-instructions).
    Returns the LINE to speak (not a prompt). Lazy-imports openai; safe-fails to ''."""
    global _OAI_CLIENT
    try:
        if _OAI_CLIENT is None:
            from openai import AsyncOpenAI
            _OAI_CLIENT = AsyncOpenAI()  # uses OPENAI_API_KEY (already required on the worker)
        r = await _OAI_CLIENT.chat.completions.create(
            model=os.getenv("NOVA_OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens, temperature=0.8)
        return (r.choices[0].message.content or "").strip().strip('"')
    except Exception as e:
        logger.error(f"[llm_line] {e}")
        return ""


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
        await _nova_say(session, line)   # EVI-safe (was session.say → silent on EVI)
        logger.info(f"[react/bank] {event_name} streak={state.ctx.streak} → '{line}'")
        return

    # Tier 2: milestone — a GROUNDED line (names the real move), never a meta-prompt.
    # (EVI can't generation-from-instructions; speaking a real bank line keeps it sharp + safe.
    #  Smart-LLM-on-past-event is the next layer.)
    move = state.ctx.current_move
    base = personality.pick_phrase("hit", max(state.ctx.streak, 5), state.ctx.name) or "yes!!"
    line = f"{move}! {base}" if move else base
    await _nova_say(session, line)
    logger.info(f"[react/milestone] {event_name} streak={state.ctx.streak} → '{line}'")


async def _speak_dance_intro(session: AgentSession, state: NovaSessionState, agent: "NovaAgent"):
    """Phase transitioned to dance — say ONE hype line over the countdown."""
    import random
    await state.pace.acquire()
    line = random.choice(["okay — let's GO!", "here we go!", "ready? GO!", "let's DANCE!", "show me what you got!"])
    await _nova_say(session, line)


async def _speak_goodbye(session: AgentSession, state: NovaSessionState, agent: "NovaAgent"):
    """Phase transitioned to goodbye — warm wrap-up."""
    await state.pace.acquire()
    memory.store.increment_sessions(state.kid_id)
    nm = (state.ctx.name + "! ") if state.ctx.name else ""
    line = (f"{nm}you did SO good — same time tomorrow?" if state.ctx.max_streak >= 3
            else f"{nm}that was fun — come back and dance with me?")
    await _nova_say(session, line)


async def _drop_in_observation(session: AgentSession, state: NovaSessionState, observation: str, agent: "NovaAgent"):
    """Drop in vision observation naturally."""
    await state.pace.acquire()
    await _nova_say(session, f"ohh... {observation}!")


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
        name = fp.claim(text, state.pace._is_speaking) if fp else None
        if name:
            state.bump("fillers")
            asyncio.create_task(fp.fire(session, name))
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

    while state.active:
        await asyncio.sleep(2)  # check every couple seconds

        # FIX: stop once the session has torn down — was firing nudges after
        # close ("[idle] nudge failed: AgentSession isn't running").
        if not state.active:
            return

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
            # Expected if the session is tearing down between our active-check and
            # say() — downgraded from error so it doesn't look like a real fault.
            logger.warning(f"[idle] nudge skipped (session ending): {e}")
            return


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
# MOVE-PLAY GAME — worker-driven loop (6-10), flexible length, 5-min cap
# ────────────────────────────────────────────────────────────────────────
async def _request_vision_and_wait(room: rtc.Room, state: NovaSessionState,
                                   timeout: float = 6.0) -> Optional[str]:
    """Ask the browser for ONE webcam frame and await the observation."""
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()
    state.vision_waiter = fut
    try:
        msg = json.dumps({"kind": "request-vision"})
        await room.local_participant.publish_data(msg.encode("utf-8"), reliable=True)
        logger.info("[game] vision requested for current move")
    except Exception as e:
        logger.error(f"[game] vision request failed: {e}")
        return None
    try:
        return await asyncio.wait_for(fut, timeout=timeout)
    except asyncio.TimeoutError:
        logger.info("[game] vision timed out — reacting blind")
        return None
    finally:
        if state.vision_waiter is fut:
            state.vision_waiter = None


async def _warm_llm(llm_instance):
    """Pre-open the OpenAI connection during startup so the FIRST kid reply isn't
    ~3s (cold TLS/pool) — warm calls are ~1.4s. Uses the SAME llm_instance the
    session uses, so the opened connection is reused. Fully guarded: any API
    mismatch just skips warming and never affects the session."""
    try:
        from livekit.agents import llm as _llm
        try:
            cc = _llm.ChatContext.empty()
        except Exception:
            cc = _llm.ChatContext()
        try:
            cc.add_message(role="user", content="hi")
        except Exception:
            pass
        stream = llm_instance.chat(chat_ctx=cc)
        async for _ in stream:
            break  # first token → connection is warm
        try:
            await stream.aclose()
        except Exception:
            pass
        logger.info("[warm] LLM connection pre-warmed (first reply will be fast)")
    except Exception as e:
        logger.info(f"[warm] LLM pre-warm skipped ({e})")


async def _wait_for_signal(state: NovaSessionState, want: str, timeout: float) -> bool:
    """Wait up to `timeout`s for the kid to give a 'yes' or 'done' signal.
    Returns True if `want` signal arrived, False on timeout/other."""
    deadline = time.time() + timeout
    state.last_kid_signal = None
    while time.time() < deadline:
        if state.game_done.is_set():
            return want == "done"
        if state.last_kid_signal == want:
            state.last_kid_signal = None
            return True
        if state.last_kid_signal and state.last_kid_signal != want:
            # opposite signal (e.g. waiting for yes, got done)
            sig = state.last_kid_signal
            state.last_kid_signal = None
            return sig == want
        await asyncio.sleep(0.3)
    return False


async def _game_say(session: AgentSession, state: NovaSessionState, line: str):
    """Speak a scripted game line, serialized through the pace gate."""
    if not state.active or state.game_done.is_set():
        return
    try:
        await state.pace.acquire()
        await session.say(line)
        logger.info(f"[game] say → '{line}'")
    except Exception as e:
        logger.warning(f"[game] say failed: {e}")


# Words that are NEVER a name (questions, fillers, commands, pronouns).
_NOT_A_NAME = {
    "yes", "no", "hi", "hey", "hello", "yeah", "yep", "nova", "okay", "ok", "um",
    "uh", "the", "a", "and", "but", "me", "my", "you", "your", "i", "im", "it",
    "what", "where", "who", "when", "why", "how", "which", "whats", "wheres",
    "are", "am", "is", "was", "can", "could", "do", "does", "did", "will",
    "hmm", "oh", "ohh", "wait", "stop", "play", "go", "name", "call", "here",
    "there", "this", "that", "good", "bad", "cool", "nice", "hungry", "tired",
}

def _extract_name(text: Optional[str]) -> Optional[str]:
    """Best-effort name pull. STRONG patterns ('my name is X') win even inside a
    question; otherwise only a clean short statement (not a question) counts."""
    if not text:
        return None
    import re
    t = text.strip()
    # Strong explicit pattern — trust it even if the sentence has a '?'
    m = re.search(r"(?:my name is|i'm|i am|call me|name's|they call me)\s+([A-Za-z][A-Za-z\-']{1,20})", t, re.I)
    if m:
        cand = m.group(1)
    else:
        # No explicit pattern: a QUESTION is never a name ("where am I?", "what?")
        if "?" in t:
            return None
        words = re.findall(r"[A-Za-z][A-Za-z\-']+", t)
        if not words or len(words) > 3:
            return None
        # the first word must not be a question/command/filler word
        if words[0].lower() in _NOT_A_NAME:
            return None
        cand = words[0]
    cand = cand.strip().capitalize()
    if cand.lower() in _NOT_A_NAME or not (1 < len(cand) <= 20):
        return None
    return cand


def _is_explicit_name(text: Optional[str]) -> bool:
    """True if the kid explicitly stated a name ('my name is X', 'call me X')."""
    if not text:
        return False
    import re
    return bool(re.search(r"(?:my name is|call me|name's|they call me|i'm |i am )", text, re.I))


async def _ambient_vision_loop(room: rtc.Room, state: NovaSessionState,
                               agent: "NovaAgent"):
    """Nova's EYES. Pulls a fresh camera observation every few seconds during
    play and stores it as the live 'RIGHT NOW YOU SEE' context, so the brain is
    NEVER blind. De-duped. Refreshes the prompt here (between turns) so vision
    stays fresh WITHOUT refreshing inside on_user_turn_completed (which would
    kill preemptive generation)."""
    interval = float(os.getenv("NOVA_VISION_INTERVAL_SEC", "4.0"))
    last = None
    await asyncio.sleep(1.0)
    while state.active and not state.game_done.is_set():
        if state.ctx.phase in ("play", "moves"):
            obs = await _request_vision_and_wait(room, state, timeout=4.5)
            if obs and not obs.startswith("(") and obs != last:
                state.ctx.observed_visual = obs
                last = obs
                logger.info(f"[vision] live → '{obs[:70]}'")
                try:
                    await agent.refresh_instructions()
                except Exception:
                    pass
        await asyncio.sleep(interval)


async def _play_heartbeat(session: AgentSession, state: NovaSessionState,
                          agent: "NovaAgent", room: rtc.Room):
    """Brain-LED pacing. Does NOT speak — it only TRIGGERS the brain to react to
    what it currently sees and keep the game moving when there's a lull. The
    brain is the single voice; this is just a heartbeat."""
    first = True
    while state.active and not state.game_done.is_set():
        await asyncio.sleep(1.2 if first else float(os.getenv("NOVA_BEAT_SEC", "4.5")))
        if state.game_done.is_set() or not state.active:
            break
        if state.pace._is_speaking:
            continue
        now = time.time()
        # kid just spoke → the normal pipeline is already replying; don't double up
        if now - state.last_kid_speech_at < 3.0:
            continue
        # give space after Nova's own last line
        if now - state.pace._last_spoke < 3.5:
            continue
        # SEATED + UPPER-BODY ONLY — the kid may be sitting, camera sees head+shoulders.
        SEATED_MOVES = ("ONLY these moves: clap, say yoo-hoo, move your head, pop a "
                        "shoulder, or touch your shoulder. NEVER jump, spin, stand, or "
                        "use hips/knees/legs.")
        if first:
            instr = (f"Start the move game NOW — call your first fun move. {SEATED_MOVES} "
                     "ONE short hype sentence.")
            first = False
        else:
            instr = ("Keep the move game going: react to what you SEE right now "
                     f"(name the body part), then call the next move. {SEATED_MOVES} "
                     "ONE short sentence.")
        try:
            await state.pace.acquire()
            if _evi_on():
                # EVI can't generate-from-instructions → make the LINE, then speak the LINE
                # (this is the INSTRUCTOR layer: Nova calls the move so the kid copies HER).
                line = await _llm_line(
                    "You are Nova, a warm kids' dance instructor. Reply with ONE short line "
                    "(<=8 words) that CALLS one fun move for the kid to copy and may name a body "
                    "part. No questions, never say 'great job/amazing/awesome'.", instr)
                await _nova_say(session, line or "copy me — clap your hands!")
            else:
                await agent.refresh_instructions()
                await session.generate_reply(instructions=instr)
            state.moves_done += 1
            state.ctx.moves_done = state.moves_done
            logger.info(f"[beat] triggered brain (beat ~{state.moves_done})")
        except Exception as e:
            logger.warning(f"[beat] failed: {e}")


async def _run_nova(session: AgentSession, state: NovaSessionState,
                    agent: "NovaAgent", room: rtc.Room):
    """Brain-LED flow: intro (name + invite) → play (heartbeat + live vision) →
    end. The brain is the ONLY voice; this sets phase, triggers beats, and
    watches signals. 5-min hard cap handled by _game_hard_cap."""
    if state.game_started:
        return
    state.game_started = True
    state.ctx.phase = "intro"
    await agent.refresh_instructions()

    # ── INTRO: capture the name from the first utterance. The brain (intro
    # persona) echoes it warmly AND invites them to play — one voice. ──
    if not state.ctx.name:
        state.kid_spoke.clear()
        try:
            await asyncio.wait_for(state.kid_spoke.wait(), timeout=20.0)
        except asyncio.TimeoutError:
            pass
        nm = _extract_name(state.last_kid_text)
        if nm:
            state.ctx.name = nm
            try:
                memory.store.update(state.kid_id, name=nm)
            except Exception:
                pass
            logger.info(f"[nova] captured name: {nm}")
            await agent.refresh_instructions()
            # push the captured name to the browser so it persists (returning-kid path)
            try:
                await room.local_participant.publish_data(
                    json.dumps({"kind": "name", "name": nm}).encode("utf-8"), reliable=True)
            except Exception:
                pass

    # ── wait for a 'yes' (the brain's intro reply does the inviting) ──
    accepted = await _wait_for_signal(state, "yes", timeout=30.0)
    if state.game_done.is_set():
        await _end_game(session, state, agent, None)
        return
    if not accepted:
        accepted = await _wait_for_signal(state, "yes", timeout=90.0)
        if not accepted or state.game_done.is_set():
            await _end_game(session, state, agent, None)
            return

    # ── PLAY: brain hosts; eyes + heartbeat on ──
    state.ctx.phase = "play"
    await agent.refresh_instructions()
    logger.info("[nova] → PLAY phase (brain-led, ambient vision on)")
    vis = asyncio.create_task(_ambient_vision_loop(room, state, agent))
    try:
        await _play_heartbeat(session, state, agent, room)
    finally:
        vis.cancel()

    # ── END ──
    await _end_game(session, state, agent, state.ctx.observed_visual)


async def _end_game(session: AgentSession, state: NovaSessionState,
                    agent: "NovaAgent", last_move: Optional[str]):
    """Warm exit line + mark done. Idempotent."""
    if getattr(state, "_game_ended", False):
        return
    state._game_ended = True
    state.game_done.set()
    state.ctx.phase = "goodbye"
    name = state.ctx.name or "friend"
    did = f' That {last_move} was COOL.' if last_move else ""
    instr = (
        f"The move game is wrapping up. Say a warm 2-sentence goodbye to {name}. "
        f"Mention ONE specific thing they did (e.g. \"{last_move or 'that last move'}\") "
        f"and tell them to come back whenever — you'll be here. "
        f"NEVER say great job/amazing/awesome."
    )
    try:
        await state.pace.acquire()
        if _evi_on():
            line = await _llm_line("You are Nova saying a warm goodbye to a kid. ONE short "
                "sentence: mention one specific thing they did + invite them back. Never say "
                "'great job/amazing/awesome'.", instr)
            await _nova_say(session, line or "you were so good — come dance with me again!")
        else:
            await agent.refresh_instructions()
            await session.generate_reply(instructions=instr)
        logger.info(f"[game] exit goodbye sent (last_move={last_move}){did}")
    except Exception as e:
        logger.warning(f"[game] exit failed: {e}")


async def _game_hard_cap(session: AgentSession, state: NovaSessionState, agent: "NovaAgent"):
    """Independent watchdog — force a goodbye at the 5-min cap."""
    cap_sec = float(os.getenv("NOVA_GAME_CAP_SEC", "300"))
    try:
        await asyncio.wait_for(state.game_done.wait(), timeout=cap_sec)
    except asyncio.TimeoutError:
        logger.info("[game] 5-min hard cap reached → ending")
        await _end_game(session, state, agent, None)


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

    # Phase-1 STT swap: Deepgram needs its own key. Fail loud at boot rather than
    # silently producing zero transcripts (the old "Nova can't hear me" failure).
    if not os.getenv("DEEPGRAM_API_KEY"):
        logger.error("[nova-v207] FATAL: DEEPGRAM_API_KEY missing on worker (STT is now Deepgram nova-3)")
        raise RuntimeError("DEEPGRAM_API_KEY required — add it on Render → worker → Environment")

    # BRAIN: Groq (Llama 3.3 70B, ~0.2s first token) when GROQ_API_KEY is set —
    # cuts ~0.8s off her response vs gpt-4o-mini (~1s). Falls back to gpt-4o-mini
    # if no Groq key. Groq is OpenAI-API-compatible, so we just override base_url.
    # (No max_completion_tokens kwarg — this pinned plugin version rejects it.)
    groq_key = os.getenv("GROQ_API_KEY")
    _temp = float(os.getenv("NOVA_TEMPERATURE", "0.85"))
    if groq_key:
        groq_model = os.getenv("NOVA_GROQ_MODEL", "llama-3.3-70b-versatile")
        llm_instance = openai_plugin.LLM(
            model=groq_model, temperature=_temp,
            api_key=groq_key, base_url="https://api.groq.com/openai/v1",
        )
        logger.info(f"[nova-v207] brain = GROQ {groq_model} (instant)")
    else:
        llm_instance = openai_plugin.LLM(
            model=os.getenv("NOVA_OPENAI_MODEL", "gpt-4o-mini"),
            temperature=_temp, api_key=openai_key,
        )
        logger.info(f"[nova-v207] brain = OpenAI {os.getenv('NOVA_OPENAI_MODEL', 'gpt-4o-mini')}")
    # Pre-warm the brain NOW (runs during Runway/session setup + greeting, ~5s of
    # slack) so the FIRST kid reply isn't cold (~3s → ~1.4s).
    asyncio.create_task(_warm_llm(llm_instance))
    # Runtime proof of the resolved voice config (env overrides default if set)
    logger.info(
        "[nova-v207] TTS RUNTIME = model=%s voice=%s stability=%s similarity=%s style=%s speed=%s"
        % (
            os.getenv("NOVA_TTS_MODEL", "eleven_flash_v2_5"),
            os.getenv("NOVA_VOICE_ID", "P6xfJudBtfcB1BM5ZWR7"),
            os.getenv("NOVA_VOICE_STABILITY", "0.65"),
            os.getenv("NOVA_VOICE_SIMILARITY", "0.90"),
            os.getenv("NOVA_VOICE_STYLE", "0.30"),
            os.getenv("NOVA_VOICE_SPEED", "0.92"),
        )
    )

    # Build session pipeline
    session_kwargs = dict(
        # ─────────────────────────────────────────────────────────────
        # STT: Deepgram Nova-3 (Phase-1 STT swap, Jun 17 2026).
        # Replaces both (a) the old browser SpeechRecognition path and
        # (b) the OpenAI Whisper worker STT. Server-side streaming STT —
        # ~200-300ms faster than browser endpointing, better at kid mishears.
        #
        # NOTE on the earlier "Deepgram broken" history: prior failure showed
        # [MIC-IN] subscribed but ZERO [HEAR] lines. Two plausible causes were
        # since fixed independently: (1) RoomInputOptions now forces 16kHz
        # audio (wrong sample rate silently produced nothing), (2) we now set
        # interim_results=True so transcripts stream continuously. If [HEAR]
        # lines STILL never appear after deploy, it's an account-entitlement
        # issue on nova-3 streaming → revert this block to OpenAI Whisper.
        #
        # endpointing=400ms matches the Silero VAD min_silence (0.4s) so turn
        # detection feels identical to before. Uses DEEPGRAM_API_KEY (worker env).
        # ─────────────────────────────────────────────────────────────
        stt=deepgram.STT(
            model=os.getenv("NOVA_STT_MODEL", "nova-3"),
            language=os.getenv("NOVA_STT_LANG", "en"),
            smart_format=True,
            interim_results=True,
            # `endpointing_ms` (NOT `endpointing`). LATENCY: 400→250→180ms so she
            # answers sooner after a short kid utterance ("yes", "done"). Floor ~150ms
            # (below that she cuts kids off mid-pause).
            endpointing_ms=int(os.getenv("NOVA_STT_ENDPOINTING", "180")),
            api_key=os.getenv("DEEPGRAM_API_KEY"),
        ),
        llm=llm_instance,
        tts=elevenlabs.TTS(
            # LOORA1 (cloned voice P6xfJudBtfcB1BM5ZWR7) — kid-warm presence.
            # model eleven_flash_v2_5 — verified working with this voice.
            # settings (see _voice_settings): stability 0.65 (calmer), similarity
            #   0.90 (locks character), style 0.30 (natural), speed 0.92 (8% slower,
            #   kid pace) when the installed plugin supports speed.
            voice_id=os.getenv("NOVA_VOICE_ID", "P6xfJudBtfcB1BM5ZWR7"),
            model=os.getenv("NOVA_TTS_MODEL", "eleven_flash_v2_5"),
            voice_settings=_voice_settings(),
        ),
        vad=silero.VAD.load(
            # v225: trimmed for snappy turn-taking. 0.4s of silence ends a turn
            # (was the ~0.55 default), 0.05s min speech, 0.2s prefix padding.
            min_speech_duration=0.05,
            min_silence_duration=0.3,
            prefix_padding_duration=0.2,
            # HEARING gate. 0.6 made her DEAF to quiet/cafe speech (no transcripts at
            # all). 0.45 = catch soft voices; noise robustness comes from
            # min_interruption_duration below, NOT from making her deaf.
            activation_threshold=0.45,
        ),
        # VAD-based turn detection. Require ~0.6s of SUSTAINED speech to INTERRUPT her
        # (Runway can't pause to resume a false interruption), but this does NOT gate
        # whether she hears the kid when she's silent — activation_threshold does that.
        turn_detection="vad",
        allow_interruptions=True,
        min_interruption_duration=0.6,
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

    # FIX (idle nudges after disconnect): when the kid leaves, the AgentSession
    # closes immediately, but state.active only flipped in the shutdown callback
    # ~20s later — so the idle loop kept calling session.say() and logging
    # "[idle] nudge failed: AgentSession isn't running". Flip it the INSTANT a
    # non-agent participant disconnects so the idle loop stops on its next tick.
    @ctx.room.on("participant_disconnected")
    def _on_participant_left(participant):
        ident = getattr(participant, "identity", "?")
        if not str(ident).startswith(("runway-", "agent-", "nova")):
            state.active = False
            logger.info(f"[nova-v207] kid {ident} disconnected → stopping idle loop")

    @ctx.room.on("disconnected")
    def _on_room_disconnected(*_a):
        state.active = False
        logger.info("[nova-v207] room disconnected → stopping idle loop")

    # ── HUME EVI 3 (speech-to-speech) ────────────────────────────────────
    # When USE_EVI=1, replace STT+LLM+TTS+VAD with a single Hume EVI realtime
    # model (Kora voice + Nova config). Runway still lip-syncs the session's
    # audio output, which is now EVI's voice. If anything fails to init, we keep
    # the normal Deepgram→LLM→ElevenLabs pipeline so Nova never goes dark.
    if _evi_on():
        try:
            from evi_realtime import HumeEVIRealtimeModel
            session_kwargs = {
                "llm": HumeEVIRealtimeModel(),   # reads HUME_API_KEY / HUME_SECRET_KEY / HUME_CONFIG_ID
                "allow_interruptions": True,
            }
            logger.info("[nova-evi] USE_EVI=1 → STT+LLM+TTS replaced with Hume EVI realtime (Kora)")
        except Exception as e:
            logger.exception(f"[nova-evi] EVI init failed → falling back to normal pipeline: {e}")

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
                # Feed the move-game orchestrator: pulse "kid spoke" + classify
                # the utterance as a done/yes signal so the loop can react.
                state.last_kid_text = text
                state.last_kid_speech_at = time.time()
                # Capture the name SYNCHRONOUSLY here (before the brain auto-replies)
                # so the very next reply uses the name + play-invite persona — fixes
                # the race where Nova asked for the name again.
                if state.ctx.phase in ("intro", "recognition") and (
                    not state.ctx.name or _is_explicit_name(text)
                ):
                    nm = _extract_name(text)
                    if nm and nm != state.ctx.name:
                        state.ctx.name = nm
                        try:
                            memory.store.update(state.kid_id, name=nm)
                        except Exception:
                            pass
                        logger.info(f"[nova] captured name (hook): {nm}")
                sig = personality.detect_signal(text)
                if sig:
                    state.last_kid_signal = sig
                    if sig == "done":
                        logger.info("[game] kid signalled DONE")
                        state.game_done.set()
                try:
                    state.kid_spoke.set()
                except Exception:
                    pass
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
        first_line = f"ohh — {state.ctx.name}! you came BACK! ...ready to play again?"
    else:
        first_line = "hi! I'm Nova — your magic movement friend! ...what's your name?"

    # Cinematic reveal: wait until the browser signals it has revealed Nova
    # (client-ready) before greeting — so she greets AFTER she appears, not behind
    # the overlay. 8s fallback so she never stays silent if the signal is missed.
    try:
        await asyncio.wait_for(state.client_ready.wait(), timeout=12.0)
        logger.info("[nova-v207] greeting released by client-ready")
    except asyncio.TimeoutError:
        logger.info("[nova-v207] greeting released by 12s fallback (no client-ready)")

    try:
        await _nova_say(session, first_line)   # EVI-safe greeting (raw session.say() fails under EVI → she was silent → blank reveal)
        logger.info(f"[nova-v207] step 8: GREETING SENT (hardcoded): '{first_line}'")
    except Exception as e:
        logger.error(f"[nova-v207] hardcoded greeting failed: {e}")

    # MOVE-PLAY GAME (default ON). The game drives its own vision (per move) and
    # its own nudges/cap, so the legacy one-shot vision loop + idle loop are NOT
    # started in game mode. Toggle off with NOVA_GAME_MODE=0 to get the old flow.
    if os.getenv("NOVA_GAME_MODE", "1") == "1":
        logger.info("[nova] brain-led move game ON — starting flow")
        asyncio.create_task(_run_nova(session, state, agent, ctx.room))
        asyncio.create_task(_game_hard_cap(session, state, agent))
    else:
        # Kick off vision request in background
        asyncio.create_task(_vision_trigger_loop(ctx.room, state))
        # Gentle idle engagement — soft presence if the child goes quiet
        asyncio.create_task(_idle_watch_loop(session, state))

    # Per-session AUTO-SUMMARY: logged once when the session tears down, so EVERY
    # session self-reports its headline metrics into the log stream (option A).
    async def _log_session_summary():
        state.active = False  # stop the idle watch loop (no ghost nudges post-close)
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
