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
                # V2V STAGES 1+5: the ears door follows the phase — open for
                # conversation, CLOSED the moment a song plays (she must never
                # listen to the music), open again for the goodbye.
                _m = getattr(self, "_evi_model", None)
                if _m is not None:
                    _m._ears_open = new_phase in ("recognition", "goodbye")
                    logger.info(f"[EARS] door {'OPEN' if _m._ears_open else 'CLOSED'} (phase {new_phase})")
                # REAL-TIME AWARENESS (2026-07-06): her brain learns the phase changed
                # THE INSTANT it happens — no more finishing old-phase thoughts late.
                _model = getattr(self, "_evi_model", None)
                if _model is not None and hasattr(_model, "push_context"):
                    _now_note = {
                        "dance": "(RIGHT NOW: the song just started and they are DANCING — "
                                 "drop any earlier thread instantly; only short in-the-moment reactions)",
                        "goodbye": "(RIGHT NOW: the song just ended — it is goodbye time; "
                                   "no game talk, wrap warmly)",
                        "recognition": "(RIGHT NOW: back at the welcome screen)",
                    }.get(new_phase)
                    if _now_note:
                        _model.push_context(_now_note)
                # PHASE 3: entering the song → fresh per-song voice gate (the router).
                # Kill-switch NOVA_P3=0 falls back to the July reaction path.
                if new_phase == "dance" and os.getenv("NOVA_P3", "1") == "1":
                    self.game_gate = personality.GameVoiceGate(
                        all_time_best=self.ctx.max_streak)
                    logger.info("[P3-ROUTER] gate armed for this song "
                                f"(all-time best={self.ctx.max_streak})")
                if new_phase == "goodbye":
                    # continuity gold: mid-song story resurfaces in the ending
                    g = getattr(self, "game_gate", None)
                    if g and g.deferred_topics:
                        self.ctx.deferred_topic = g.deferred_topics[0]
                        logger.info(f"[P3-ROUTER] deferred topic → ending: "
                                    f"'{self.ctx.deferred_topic[:60]}'")

        # Game events
        elif ev == "hit":
            self.ctx.hits = event.get("hits", self.ctx.hits + 1)
            self.ctx.streak = event.get("streak", self.ctx.streak + 1)
            # ENDING: remember WHICH moves really landed — the goodbye callback must
            # reference a real moment, never an invented one.
            _a = (event.get("action") or "").strip().lower()
            if _a:
                acts = getattr(self, "_hit_actions", None)
                if acts is None:
                    acts = set(); self._hit_actions = acts
                acts.add(_a)
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
            acts = getattr(self, "_hit_actions", None)
            if acts is None:
                acts = set(); self._hit_actions = acts
            acts.add("freeze")

        elif ev == "freeze_miss":
            self.ctx.last_event = "freeze_miss"
            self.ctx.streak = 0

        elif ev == "music_tick":
            self.ctx.music_sec = float(event.get("sec", 0))
            # TALK SCORE: every tick re-syncs the worker's song clock to the REAL
            # music position (browser is the source of truth — pauses/lag included).
            self._talk_t0 = time.time() - self.ctx.music_sec
            g = getattr(self, "game_gate", None)
            if g:
                g.tick(self.ctx.music_sec, time.time())

        elif ev in ("detection", "detection_lost", "detection_back"):
            # PHASE 3 edge 2: detection died / recovered mid-song → dance-along voice
            g = getattr(self, "game_gate", None)
            if g:
                g.set_detection(event.get("ok", ev == "detection_back"))
                logger.info(f"[P3-ROUTER] detection_ok={g.detection_ok} "
                            f"({'dance-along voice' if not g.detection_ok else 'full presence'})")

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
                # PHASE 1: age asked ONCE, kept FOREVER — returning kids skip the age beat
                try:
                    _mem = memory.store.get(self.kid_id)
                    _mem.add_shared_fact("age_tier", tier)
                    memory.store.save(_mem)
                    self.ctx.shared_facts["age_tier"] = tier
                except Exception as e:
                    logger.warning(f"[memory] age_tier save failed: {e}")

        elif ev == "away":
            self._kid_away = True    # ENDING edge 2: goodbye to an empty room = one soft line
            # PHASE 3 edge 1: mid-song the gate owns it (ONE warm call, song never
            # pauses) — the reaction path handles it; skip the intro-style nudge.
            if self.ctx.phase == "dance" and getattr(self, "game_gate", None):
                pass
            # PHASE 1: kid left frame+voice for ~12s → ONE warm nudge, then quiet.
            elif not getattr(self, "_away_nudged", False):
                self._away_nudged = True
                sess = getattr(self, "session", None)
                if sess:
                    asyncio.create_task(_nova_say(sess, "I'm right here when you're ready!"))
                logger.info("[away] one nudge sent → quiet waiting")

        elif ev == "back":
            # kid returned → warm re-greet; the flow resumes from conversation context.
            # NOTE: under EVI, instructions= is spoken VERBATIM — final line only.
            self._away_nudged = False
            self._kid_away = False
            # PHASE 3 edge 1: mid-song return = SEAMLESS resume, zero comment.
            if self.ctx.phase == "dance" and getattr(self, "game_gate", None):
                logger.info("[P3-ROUTER] kid back in frame → seamless resume (silent)")
            else:
                sess = getattr(self, "session", None)
                if sess:
                    asyncio.create_task(_nova_say(sess, "there you are!"))
                logger.info("[away] kid returned → resuming the beat")

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
            # PHASE 3: an open cue window is a hard no-speak zone (kid concentrating).
            g = getattr(self, "game_gate", None)
            if g:
                g.cue_opened(time.time())

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
        txt = None
        try:
            tc = getattr(new_message, "text_content", None)
            txt = tc() if callable(tc) else tc
        except Exception:
            txt = None
        # PHASE 3: mid-song kid speech NEVER gets a full chatbot reply. The gate
        # routes it (question → one line, story → "mm!" + after-song continuity)
        # and we cancel the pipeline's automatic generation via StopResponse.
        if (txt and getattr(self.state.ctx, "phase", "") == "dance"
                and getattr(self.state, "game_gate", None) is not None
                and _StopResponse is not None):
            sess = getattr(self.state, "session", None)
            if sess:
                asyncio.create_task(
                    _handle_dance_mic_text(sess, self.state, self, txt))
            raise _StopResponse()
        # GAME-PUSH (voice path): the kid's SPOKEN "let's dance / yes / ready" arrives HERE
        # (Deepgram), not via user-said — hook the same escort. Her natural LLM reply still
        # plays (it's already hype); we just open the picker under it. No extra spoken line.
        try:
            if txt and getattr(self.state.ctx, "phase", "") == "recognition" and _wants_to_start(txt):
                logger.info(f"[GAME-PUSH] start intent heard (voice) in '{str(txt)[:40]}'")
                asyncio.create_task(_push_to_game(self.state, getattr(self.state, "session", None), ""))
        except Exception as e:
            logger.warning(f"[GAME-PUSH] voice hook error: {e}")
        try:
            fp = getattr(self.state, "filler", None)
            sess = getattr(self.state, "session", None)
            if fp and sess:
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
    state.room = room   # so _push_to_game can send go-picker to the browser
    # VOICE-SILENCE DEBUG (2026-07-09): let the voice adapter announce its
    # generation lifecycle (gen-start / gen-done / gen-error) into the room
    _vm = getattr(state, "_evi_model", None)
    if _vm is not None and hasattr(_vm, "_announce"):
        def _diag_announce(payload):
            asyncio.create_task(room.local_participant.publish_data(
                json.dumps(payload).encode("utf-8"), reliable=True))
        _vm._announce = _diag_announce
    if _v2v_on():
        async def _announce_v2v():
            await asyncio.sleep(1.0)
            try:
                await room.local_participant.publish_data(
                    json.dumps({"kind": "v2v", "on": True}).encode("utf-8"), reliable=True)
                logger.info("[V2V] announced to browser — browser-STT conversation path gated OFF")
            except Exception:
                pass
        asyncio.create_task(_announce_v2v())

    # VOICE-SILENCE DEBUG (2026-07-09): the worker's env is invisible from
    # outside (separate Render service from the web API) — announce the voice
    # flags + key fingerprints into the room so a probe can read the truth.
    # Fingerprints only (sha256[:10]), never the keys themselves.
    async def _announce_env_diag():
        await asyncio.sleep(1.2)
        try:
            import hashlib as _hl
            def _fp(v):
                return _hl.sha256((v or "").encode()).hexdigest()[:10]
            await room.local_participant.publish_data(json.dumps({
                "kind": "env-diag",
                "USE_GEMINI": os.getenv("USE_GEMINI", ""),
                "USE_EVI": os.getenv("USE_EVI", ""),
                "NOVA_V2V": os.getenv("NOVA_V2V", ""),
                "NOVA_GEMINI_MODEL": os.getenv("NOVA_GEMINI_MODEL", "(default)"),
                "gemini_key_fp": _fp(os.getenv("GEMINI_API_KEY")),
                "google_key_fp": _fp(os.getenv("GOOGLE_API_KEY")),
                "hume_key_set": bool(os.getenv("HUME_API_KEY")),
                "lemon_key_set": bool(_lemon_key()),
                "avatar_pick": os.getenv("NOVA_AVATAR", "lemonslice"),
            }).encode("utf-8"), reliable=True)
        except Exception:
            logger.exception("[env-diag] announce failed")
    asyncio.create_task(_announce_env_diag())

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

                # TALK SCORE (2026-07-05): the browser announces the song start —
                # arm the per-song score; it owns the in-game voice from here.
                if event.get("event") == "song_start":
                    _song = (event.get("song") or "").strip()
                    state._talk_t0 = time.time() - float(event.get("sec", 0) or 0)
                    # fresh round: ending trackers reset (play-again replays in-session)
                    state._goodbye_ran = False
                    state._goodbye_skip = False
                    state._hit_actions = set()
                    state._talk_score_song = None   # allow re-arm for the same song
                    state.ctx.hits = 0
                    state.ctx.streak = 0
                    logger.info(f"[TALK-SCORE] song_start '{_song}' (sec={event.get('sec', 0)})")
                    asyncio.create_task(_run_talk_score(session, state, _song))

                # ENDING: play-again = pure delight, goodbye skipped, deposit already saved
                if event.get("event") == "play_again":
                    state._goodbye_skip = True
                    logger.info("[ENDING] play-again → goodbye skipped (deposit stays)")
                    asyncio.create_task(_nova_say(session, "AGAIN?! okay okay—"))

                # If a hit/miss happened during DANCE, react immediately.
                # PHASE 3: the router also handles song moments + edge events.
                if state.ctx.phase == "dance" and event.get("event") in (
                    "hit", "miss", "first_hit", "freeze_hit", "freeze_miss",
                    "music_moment", "section", "rep_done", "free_fun", "idle",
                    "second_person", "singing", "mic_text", "away",
                ):
                    if getattr(state, "_talk_score_active", False):
                        # the score is the conductor: real hits get the per-song echo;
                        # routine router chatter is suppressed (kid speech still routes).
                        if event.get("event") in ("hit", "first_hit", "clean_hit", "freeze_hit"):
                            asyncio.create_task(_talk_echo(session, state, event.get("event")))
                        elif event.get("event") in ("mic_text", "singing", "second_person", "away"):
                            asyncio.create_task(_react_to_event(session, state, agent, event))
                    else:
                        asyncio.create_task(_react_to_event(session, state, agent, event))

                # POINT 4 (2026-07-01): INTRO try-move — during the intro the browser
                # detects the move Nova asked for (e.g. a clap), lights the kid's hands,
                # and sends this. She reacts to the REAL move (dance-phase react above
                # only fires in "dance", so the intro needs its own hook).
                if event.get("event") == "try_move" and state.ctx.phase in ("recognition", "intro", "play"):
                    _act = (event.get("action") or "").lower()
                    _eng = getattr(state, "_turn_engine", None)
                    if _eng is not None and _eng.offer("detection", _act):
                        pass   # the beat speaks the WOW
                    elif not getattr(state, "_challenge_active", None) and not getattr(state, "_turn_engine", None):
                        asyncio.create_task(
                            _react_to_intro_move(session, state, agent, event.get("action")))

                # Phase transition to goodbye → speak final goodbye
                if event.get("event") == "phase" and event.get("phase") == "goodbye":
                    asyncio.create_task(_speak_goodbye(session, state, agent))

                # PHASE 2 TRANSITION bridge — the browser drives the button→game
                # bridge and asks for ONE beat at a time (hype/tip/framing/framed/
                # slow/switch/fail/dancealong). It owns the budget + ready-gate; we
                # just speak the beat's line. See personality.transition_line.
                if event.get("event") == "bridge":
                    asyncio.create_task(
                        _speak_bridge(session, state, agent,
                                      event.get("beat"), event.get("song")))

                # Phase transition to dance → fire the GO-LINE ("here we GO!"), the
                # final beat of the transition. The browser waits for her to start
                # this line, then starts the MP4 (face hidden). This replaces the old
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
                    # COMMERCIAL-INTRO A.3: vision is a WHISPER — her awareness,
                    # never a turn. She weaves it in when she naturally speaks.
                    elif _v2v_on() and getattr(state, "_evi_model", None) is not None:
                        _whisper(state, "VISION", obs)
                    # Legacy one-shot drop-in (non-EVI only).
                    elif not state.game_started and not state.vision_fired:
                        state.vision_fired = True
                        asyncio.create_task(_drop_in_observation(session, state, obs, agent))

            elif kind == "presence":
                # COMMERCIAL-INTRO C.3 (the air rule): the browser reports frame
                # presence transitions. Absent → ONE soft call, then QUIET waiting
                # (zero talking to the air). Partial signs → one more gentle try.
                # Return → whisper a warm resume at the same beat.
                present = bool(msg.get("present"))
                partial = bool(msg.get("partial"))
                was = getattr(state, "_kid_present", True)
                state._kid_present = present
                if not present and was:
                    state._air_calls = 0
                    logger.info("[AIR] nobody in frame — quiet waiting state armed")

                    async def _air_call():
                        await asyncio.sleep(4.0)   # not a blip — really gone
                        if getattr(state, "_kid_present", True) or not state.active:
                            return
                        if state.ctx.phase not in ("intro", "recognition", "play"):
                            return
                        if getattr(state, "_air_calls", 0) >= 1:
                            return
                        state._air_calls = 1
                        await _nova_say(session, "…you there, friend?")
                        logger.info("[AIR] one soft call made → QUIET (watching)")
                    asyncio.create_task(_air_call())
                elif not present and partial and getattr(state, "_air_calls", 0) == 1:
                    state._air_calls = 2
                    logger.info("[AIR] partial signs of someone → one more gentle try")
                    asyncio.create_task(_nova_say(session, "…you there, friend?"))
                elif present and not was:
                    state._air_calls = 0
                    _beat = getattr(state, "_current_beat_name", None)
                    _whisper(state, "EVENTS",
                             f"they're BACK in frame — resume warmly right where you were"
                             + (f" (beat '{_beat}')" if _beat else ""))
                    logger.info("[AIR] person returned → warm resume whispered")

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

            elif kind == "clip-ack":
                _ev = msg.get("ev")
                if _ev == "duet-pause":
                    # ONE-BEAT-ONE-MOUTH INVARIANT: a clip pausing under her live voice
                    # in a conversation beat means TWO mouths were ordered — that is the
                    # double-voice bug, and it must scream in the logs, never hide.
                    logger.error(f"[ONE-MOUTH] clip '{msg.get('id')}' paused under live EVI voice "
                                 f"(beat {msg.get('beat')}) — two mouths were ordered")
                elif _ev == "start":
                    # dead-air detector: a playing clip IS the voice in the room
                    state._clip_playing = True
                elif _ev in ("end", "error", "cancelled"):
                    state._clip_playing = False
                    state._last_clip_end_at = time.time()

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

            elif kind == "reveal-now":
                # INTRO-FINAL: THE one greeting trigger. Browser starts the calm fade →
                # sends reveal-now → we speak. No other path may fire the greeting.
                # Dedupe here + in the model; ack so the browser stops resending.
                async def _reveal_greet():
                    state.client_ready.set()   # stop the nova-ready announcer too
                    try:
                        await room.local_participant.publish_data(
                            json.dumps({"kind": "reveal-ack"}).encode("utf-8"), reliable=True)
                    except Exception:
                        pass
                    if getattr(state, "_reveal_greeted", False):
                        return
                    state._reveal_greeted = True
                    llm_obj = getattr(state, "_evi_model", None) or getattr(session, "llm", None)
                    # V2V: the ears open HERE — the reveal is the start of the
                    # conversation; anything the mic caught before it is not a turn.
                    if llm_obj is not None and _v2v_on():
                        llm_obj._ears_open = True
                        logger.info("[EARS] door OPEN (reveal — conversation starts)")
                    # CLIP GREETING (root fix): first meetings get the pre-rendered hello
                    # INSTANTLY (the 25-45s cold gen was the single worst latency wart).
                    # Returning kids keep the live EVI greet (deposit opener needs her brain).
                    _returning = bool(state.ctx.name and state.ctx.sessions_before >= 1)
                    if (not _returning and _CLIP_INDEX and os.getenv("NOVA_CLIPS", "1") == "1"
                            and not getattr(state, "_direct_game", None)):
                        if hasattr(llm_obj, "_greet_fired"):
                            llm_obj._greet_fired = True   # EVI greet suppressed — clip owns it
                        logger.info("[REVEAL] greeting via CLIP (instant)")
                        asyncio.create_task(_nova_say(
                            session, "hi! I'm Nova — your magic friend! …what's your name?"))
                    elif hasattr(llm_obj, "fire_greeting"):
                        _live = bool(getattr(llm_obj, "connected_evt", None) and llm_obj.connected_evt.is_set())
                        logger.info(f"[REVEAL] greeting fired → EVI (ws_live={_live})")
                        if not _live:
                            logger.error("[REVEAL] EVI ws NOT LIVE at greeting time — nudge is queued; she may be mute until connect")
                        llm_obj.fire_greeting()
                        # MUTE GUARD: if she hasn't spoken 12s after firing, retry ONCE, loudly.
                        # 12s not 7 (Hume cold first-gen ran 28s live) + a speaking-state check —
                        # the 7s guard re-fired the nudge in the same second her first word landed,
                        # and the duplicate became a phantom user turn ("ohh — cool name!" to nobody).
                        async def _mute_guard():
                            t0 = time.time()
                            await asyncio.sleep(12.0)
                            if getattr(state, "_last_nova_at", 0) >= t0 or getattr(state, "_is_speaking", False):
                                return   # she spoke / is mid-word — all good
                            logger.error("[REVEAL] NO SPEECH 12s after greeting — retrying fire once (self-guarding)")
                            try:
                                llm_obj.fire_greeting(retry=True)
                            except Exception as ge:
                                logger.error(f"[REVEAL] greeting retry failed: {ge}")
                            await asyncio.sleep(10.0)
                            if getattr(state, "_last_nova_at", 0) < t0 and not getattr(state, "_is_speaking", False):
                                logger.error("[REVEAL] STILL MUTE after retry — EVI session dead (check W0112/connect errors above)")
                        asyncio.create_task(_mute_guard())
                    else:
                        logger.info("[REVEAL] greeting fired → non-EVI say()")
                        # non-EVI path: speak the deterministic opener
                        line = getattr(state, "_first_line", None) or "hey there… I'm Nova. …I can see you, you know."
                        await _nova_say(session, line)
                    asyncio.create_task(_silence_driver(state, session))   # FIX 4: she LEADS on silence
                asyncio.create_task(_reveal_greet())

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
    # HUME-ONLY (2026-07-08): Deepgram STT is deleted, so there is no fallback
    # pipeline anymore — the entrypoint guard REQUIRES this to return True.
    # Render env must have NOVA_FORCE_ELEVENLABS=0 and USE_EVI=1.
    if os.getenv("NOVA_FORCE_ELEVENLABS", "1") == "1":
        return False
    return os.getenv("USE_EVI", "").lower() in ("1", "true", "yes", "on")


# ── CLIP VOICE (2026-07-06, the root fix): every FIXED line is pre-rendered in the
# real Kora voice and shipped to the browser. _nova_say looks the exact text up in
# the clip index — hit = the browser plays it in MILLISECONDS (play-clip packet);
# miss (dynamic text) = the EVI live path as before. One change point, every
# scripted call site upgraded at once. Kill-switch: NOVA_CLIPS=0.
_CLIP_INDEX: dict = {}   # exact text -> (clip id, duration seconds)
try:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips-manifest.json"),
              encoding="utf-8") as _cf:
        _CLIP_INDEX = {v["text"].strip(): (k, float(v.get("dur", 2.5)))
                       for k, v in json.load(_cf).items()}
except Exception as _ce:
    _CLIP_INDEX = {}
if _CLIP_INDEX:
    logging.getLogger("nova-v207").info(f"[CLIP] index loaded: {len(_CLIP_INDEX)} lines clip-backed")


async def _nova_say(session: AgentSession, line: str):
    """Speak an EXACT line — THE one speech path.
    1) CLIP FIRST: exact text found in the pre-rendered pack → browser plays the real
       Kora audio instantly (the latency root-fix).
    2) EVI fallback: dynamic text → generate_reply(instructions=) speaks it verbatim.
    Never a meta-prompt (that's the 'read the AI instructions aloud' fix)."""
    if not line:
        return False
    st = getattr(session, "_nova_state", None)
    hit = _CLIP_INDEX.get(line.strip()) if os.getenv("NOVA_CLIPS", "1") == "1" else None
    cid, cdur = (hit if hit else (None, 0.0))
    if cid and st is not None and getattr(st, "room", None) is not None:
        try:
            # V2V NO-SELF-HEARING (spec Stage 1, was missing): her clip voice plays on
            # the kid's SPEAKERS — with the door open the mic feeds it straight back to
            # EVI, which answers "the other voice" = the live double-voice bug. Ears
            # CLOSED for the clip's exact duration, and any in-flight EVI speech is cut
            # first (one mouth). Door restores to the phase's base state afterwards.
            _model = getattr(st, "_evi_model", None)
            if _model is not None and _v2v_on():
                _model._ears_open = False
                # ONE BEAT = ONE MOUTH: this beat is clip-owned — EVI generation is
                # HELD until the clip ends AND kid input arrives (or the worker
                # explicitly orders her next line). Suppression enforced in the bridge.
                _model._mouth_hold = True
                _model._clip_playing = True
                try:
                    _r = session.interrupt()
                    if asyncio.iscoroutine(_r):
                        await _r
                except Exception:
                    pass
            await st.room.local_participant.publish_data(
                json.dumps({"kind": "play-clip", "id": cid}).encode("utf-8"), reliable=True)
            # CONTEXT SYNC: her brain must know what her mouth just said (clips bypass
            # EVI) — otherwise the kid replies to a question she "never asked".
            recent = getattr(st, "_clip_recent", None) or []
            recent.append(line.strip()); st._clip_recent = recent[-3:]
            model = getattr(st, "_evi_model", None)
            if model is not None and hasattr(model, "push_context"):
                ctx_txt = "You just said out loud (the game system spoke these for you): " +                           " | ".join(f'"{x}"' for x in st._clip_recent)
                model.push_context(ctx_txt)
            st._order_n = getattr(st, "_order_n", 0) + 1
            logger.info(f"[ORDER] #{st._order_n} clip {cid}")
            logger.info(f"[CLIP] ▶ {cid} ({cdur}s)  ('{line[:50]}')")
            # HER MOUTH IS BUSY for the clip's real duration (2026-07-07: clips return
            # instantly, so the challenge armed mid-ask, acks overlapped the greeting,
            # transitions rushed). Holding here restores natural pacing everywhere:
            # ask-first arming, goodbye beats, input windows.
            await asyncio.sleep(cdur + 0.25)
            st._last_nova_at = time.time()
            if _model is not None and _v2v_on():
                _model._clip_playing = False
                _open = getattr(st.ctx, "phase", "intro") in ("intro", "recognition", "goodbye")
                _model._ears_open = _open
                logger.info(f"[EARS] clip done → door {'OPEN' if _open else 'CLOSED'} (phase {getattr(st.ctx, 'phase', '?')})")
            return True
        except Exception as e:
            logger.warning(f"[CLIP] publish failed → EVI fallback: {e}")
    try:
        if _gemini_on():
            # Gemini treats instructions as guidance, not a script — pin it verbatim
            await session.generate_reply(
                instructions=f'Say exactly this, word for word, nothing more: "{line}"')
        elif _evi_on():
            await session.generate_reply(instructions=line)
        else:
            await session.say(line)
        st2 = getattr(session, "_nova_state", None)
        if st2 is not None:
            st2._order_n = getattr(st2, "_order_n", 0) + 1
            logger.info(f"[ORDER] #{st2._order_n} directed-line (EVI)")
        logger.info(f"[SAY] '{line[:60]}'")
        return True
    except Exception as e:
        logger.error(f"[SAY] failed: {e}")
        return False


# POINT 4 (2026-07-01): Nova reacts to the kid's REAL move during the INTRO try-move.
# The browser detects it + lights the hands, then sends a game-event {try_move}. Bank
# lines keep it INSTANT and on-persona (name the body part, no lazy praise).
_INTRO_MOVE_REACT = {
    "clap":  ["whoa — I SAW that clap! those HANDS!", "yesss — your hands SNAPPED together!", "ooh, that clap was SHARP!"],
    "hands": ["whoa — I SAW those hands!", "yesss — your hands lit UP!"],
    "right": ["look at that RIGHT hand — UP it went!", "whoa — right hand SHOT up!"],
    "left":  ["that LEFT hand, up high — I see you!", "ooh — left hand UP!"],
    "both":  ["BOTH hands UP — big energy!", "whoa — both arms in the air!"],
    "head":  ["look at that head move — I see you!", "ooh, you moved your HEAD!"],
    "shoulder": ["that SHOULDER popped — I saw it!", "ooh — shoulder UP! you found the light!"],
}


async def _react_to_intro_move(session: AgentSession, state: NovaSessionState,
                               agent: "NovaAgent", action: Optional[str]):
    """Speak ONE instant reaction to the move the kid actually did in the intro.
    One-shot per action so a sticky/repeated detection can't make her spam it."""
    act = (action or "clap").lower()
    seen = getattr(state, "_intro_moves_reacted", None)
    if seen is None:
        seen = set()
        state._intro_moves_reacted = seen
    if act in seen:
        return
    seen.add(act)
    line = random.choice(_INTRO_MOVE_REACT.get(act, _INTRO_MOVE_REACT["clap"]))
    try:
        await state.pace.acquire()
        await _nova_say(session, line)
        logger.info(f"[intro-move] reacted to '{act}': {line}")
    except Exception as e:
        logger.warning(f"[intro-move] react failed: {e}")


# ════════════════════════════════════════════════════════════════════
# SCRIPTED MOVEMENT CHALLENGE (2026-07-04, Refael's design)
# ────────────────────────────────────────────────────────────────────
# The LLM does NOT choose or time the challenge — sync was impossible
# (light late/wrong joint, 4s blind spot, premature reactions). The
# WORKER runs a fixed script: max 2 pre-made moves, light locked to the
# EXACT joint the moment the cue line is spoken, neutral filler while
# waiting, pre-made WOW only when the browser's detection confirms.
# ════════════════════════════════════════════════════════════════════
# COMMERCIAL-INTRO Part C.4 (2026-07-08): the 2-CUE LIGHT CHALLENGE is a CHAIN —
# the second cue IS the instant reaction to the first hit ("it JUMPED to your
# HAND!"), so a hit continues the chain instead of ending it. Every line here is
# a pre-rendered clip (≤0.5s from detection); "wow": None = the next cue is the wow.
INTRO_CHALLENGE = [
    {"action": "shoulder", "joint": "right_shoulder",
     "cue":    "okay… see that sparkle? can you nudge that RIGHT shoulder?",
     "filler": "let's see it… give that shoulder a little push!",
     "wow":    None},   # the hit's reward is the INSTANT jump cue below
    {"action": "left", "joint": "left_wrist",
     "cue":    "YES!! okay okay — now look, it JUMPED to your HAND!",
     "filler": "give that hand a little shake — go!",
     "wow":    "you GOT it!!"},
]
_CHALLENGE_CLOSE_WIN  = "ready to dance? push the big button — or say 'let's start'!"
_CHALLENGE_CLOSE_MISS = "I LOVE that energy! ready to dance? push the big button!"


async def _run_intro_challenge(session: AgentSession, state: NovaSessionState,
                               room: rtc.Room):
    """The fixed, fully-synced challenge. One run per session."""
    if getattr(state, "_challenge_ran", False):
        return
    state._challenge_ran = True
    # BREATH (2026-07-06): never storm the kid — the name ack / her current line
    # finishes first, then ONE human beat, THEN the first cue.
    for _ in range(40):
        if not getattr(state, "_is_speaking", False) and not getattr(state, "_say_inflight", None):
            break
        await asyncio.sleep(0.25)
    await asyncio.sleep(2.0)
    hit = False
    for mv in INTRO_CHALLENGE:
        # ABORT if the intro is over (kid pressed the button / game started) — the
        # challenge must never talk over a running game (live 2026-07-04)
        if state.game_done.is_set() or state.ctx.phase not in ("intro", "recognition"):
            logger.info("[CHALLENGE] aborted — intro is over (phase moved on)")
            state._challenge_active = None
            return
        state._challenge_active = mv["action"]
        state._challenge_done = asyncio.Event()
        # 1) VOICE FIRST (2026-07-06 live: kid was already moving → detector confirmed
        # BEFORE she even asked → "WOW" with no question = conversation broken).
        # She asks; the light + detection arm only once the ask has been SPOKEN.
        await _nova_say(session, mv["cue"])
        try:
            await room.local_participant.publish_data(
                json.dumps({"kind": "cue-part", "part": mv["action"], "joint": mv["joint"]}).encode("utf-8"),
                reliable=True)
        except Exception as e:
            logger.warning(f"[CHALLENGE] cue publish failed: {e}")
        state._last_cue_part = (mv["action"], time.time())   # keep _scan_nova_line dedupe in sync
        logger.info(f"[CHALLENGE] ask spoken → light + detection armed on {mv['joint']}")
        # 3) wait; neutral filler at ~4.5s (she does NOT know yet — never claims success)
        try:
            await asyncio.wait_for(state._challenge_done.wait(), timeout=4.5)
        except asyncio.TimeoutError:
            pass
        if state.ctx.phase not in ("intro", "recognition"):
            logger.info("[CHALLENGE] aborted mid-move — game started")
            state._challenge_active = None
            return
        if not state._challenge_done.is_set():
            # NEVER TALK TO THE AIR: a kid who has given ZERO input (no voice, no
            # typing) gets the cue + the light and then QUIET — no filler at nobody.
            if getattr(state, "_last_kid_at", 0) or getattr(state, "last_kid_text", None):
                await _nova_say(session, mv["filler"])
            try:
                await asyncio.wait_for(state._challenge_done.wait(), timeout=4.0)
            except asyncio.TimeoutError:
                pass
        if state.ctx.phase not in ("intro", "recognition"):
            logger.info("[CHALLENGE] aborted mid-move — game started")
            state._challenge_active = None
            return
        # 4) confirmed → pre-made WOW; not → next scripted move
        if state._challenge_done.is_set():
            logger.info(f"[CHALLENGE] {mv['action']} CONFIRMED by detection → WOW")
            await _nova_say(session, mv["wow"])
            hit = True
            break
        logger.info(f"[CHALLENGE] {mv['action']} not detected in window → next move")
    state._challenge_active = None
    seen = getattr(state, "_intro_moves_reacted", None) or set()
    seen.add("scripted")
    state._intro_moves_reacted = seen        # unlocks the kid's 'yes' → game
    state._dance_invited = True
    await _nova_say(session, _CHALLENGE_CLOSE_WIN if hit else _CHALLENGE_CLOSE_MISS)
    logger.info(f"[CHALLENGE] done (hit={hit}) → dance invite out")


# ════════════════════════════════════════════════════════════════════
# PRODUCER = WHISPERER (COMMERCIAL-INTRO-GAME, 2026-07-08, Part A).
# The backend NEVER speaks and NEVER creates turns. It feeds her awareness
# through the Layer-4 "RIGHT NOW" channel (Hume session_settings.context —
# structurally turn-less, so she cannot "answer" the producer). Channels:
# VISION / MEMORY / STAGE / EVENTS. generate_reply is BANNED in
# conversational beats (lopo.md: 4 timeouts = 15-40s dead-air holes) —
# when her natural flow can't cover a moment (silent kid = no turn
# exists), an instant CLIP covers it instead.
# ════════════════════════════════════════════════════════════════════
def _whisper(state, channel: str, text: str) -> bool:
    """Producer whisper → her awareness. No turn, no voice, never audible."""
    model = getattr(state, "_evi_model", None)
    if model is None or not _v2v_on():
        return False
    try:
        ok = model.push_context(f"(RIGHT NOW — {channel}: {text})")
        logger.info(f"[WHISPER] {channel}: {text[:100]}")
        return bool(ok)
    except Exception as e:
        logger.warning(f"[WHISPER] {channel} failed: {e}")
        return False


async def _dead_air_watch(state):
    """COMMERCIAL-INTRO Part A.5/E: any conversational gap >4s where the ball is
    in NOVA's court = ERROR with cause. The kid thinking inside a listen window
    is intentional silence (badge says Listening) — NOT dead air."""
    fired_for = 0.0
    while state.active and not state.game_done.is_set():
        await asyncio.sleep(1.0)
        if state.ctx.phase not in ("intro", "recognition", "play"):
            continue
        now = time.time()
        her_last = getattr(state, "_last_nova_at", 0) or 0
        kid_last = getattr(state, "last_kid_speech_at", 0) or 0
        clip_end = getattr(state, "_last_clip_end_at", 0) or 0
        last_voice = max(her_last, kid_last, clip_end)
        if last_voice <= 0 or getattr(state, "_is_speaking", False) or getattr(state, "_clip_playing", False):
            continue
        gap = now - last_voice
        if gap <= 4.0 or last_voice == fired_for:
            continue
        # whose court? hers if the kid acted after her last word, or a say is stuck in flight
        say_stuck = getattr(state, "_say_inflight", None)
        if kid_last > max(her_last, clip_end):
            fired_for = last_voice
            logger.error(f"[DEAD-AIR] {gap:.1f}s since kid spoke, no reply started "
                         f"(beat={getattr(state, '_current_beat_name', '?')}, cause="
                         f"{'EVI generation in flight' if say_stuck else 'no reply triggered'})")
        elif say_stuck and now - say_stuck > 4.0:
            fired_for = last_voice
            logger.error(f"[DEAD-AIR] {gap:.1f}s — a line has been stuck in EVI for {now - say_stuck:.1f}s "
                         f"(beat={getattr(state, '_current_beat_name', '?')})")


# ════════════════════════════════════════════════════════════════════
# TURN ENGINE (FIX-TURN-OWNER 2026-07-07) — THE one conversation owner.
# BEAT = ask → (clip plays to its END; _nova_say holds the real duration)
#        → listen window opens → inputs resolve against THIS beat only
#        → resolve → advance. One beat at a time. She never performs an
# action she just offered — timeouts lead, answers resolve.
# ════════════════════════════════════════════════════════════════════
class TurnEngine:
    def __init__(self, session, state, room):
        self.session, self.state, self.room = session, state, room
        self.beat_seq = 0
        self.beat_name = None
        self.listening = False
        self._matcher = None
        self._result = None
        self._got = asyncio.Event()

    def new_beat(self, name):
        self.beat_seq += 1
        self.beat_name = name
        self.listening = False
        self.state._current_beat = self.beat_seq
        self.state._current_beat_name = name   # dead-air detector names the scene
        logger.info(f"[TURN] beat #{self.beat_seq} '{name}'")
        return self.beat_seq

    async def cancel_stale(self):
        """State advanced — no clip from a dead beat may reach the speaker."""
        try:
            await self.room.local_participant.publish_data(
                json.dumps({"kind": "cancel-beat", "beat": self.beat_seq - 1}).encode("utf-8"),
                reliable=True)
        except Exception:
            pass

    async def ask(self, line):
        """Speak the beat's ask. _nova_say holds for the clip's REAL duration
        (or the full EVI generation) — when this returns, the ask has been heard."""
        await _nova_say(self.session, line)

    async def brief(self, goal, fallback_line):
        """COMMERCIAL-INTRO Part A (2026-07-08): ZERO generate_reply in
        conversational beats — that path caused the 15-40s dead-air holes
        (4 timeouts in lopo.md). The producer WHISPERS the beat goal (no turn);
        if the kid just spoke, her own V2V reply carries the beat. A silent kid
        means no turn exists, so an instant CLIP covers the moment instead."""
        model = getattr(self.state, "_evi_model", None)
        if not _v2v_on() or model is None:
            await self.ask(fallback_line)
            return
        _whisper(self.state, "STAGE", f"beat '{self.beat_name}' — {goal}")
        # ONE MOUTH (2026-07-07 live run): when the previous beat was resolved by
        # her OWN hearing, EVI is already answering the kid — the whisper rides
        # that reply. Wait for it to land (begin ≥2s, done = 1.8s quiet, cap 10s).
        # FIX-TYPED-CHAT (2026-07-08): typed = spoken, ONE pipeline. A beat resolved
        # by TYPED text is a real kid turn exactly like voice — the next beat's clip
        # must NOT fire over her in-flight reply (it interrupted + mouth-held the
        # typed-name mirror: the "she never replies to text" bug).
        _res = getattr(self, "_last_resolution", None)
        kid_just_spoke = (
            (_res is not None and _res[0] in ("stt", "typed") and time.time() - _res[1] < 6.0)
            or time.time() - (getattr(self.state, "last_kid_speech_at", 0) or 0) < 3.0
        )
        if kid_just_spoke:
            _before = getattr(self.state, "_last_nova_at", 0)
            _w0 = time.time()
            while time.time() - _w0 < 14.0:
                # her reply LANDED (item added stamps _last_nova_at at turn end) → done
                if (getattr(self.state, "_last_nova_at", 0) > _before
                        and not getattr(self.state, "_is_speaking", False)):
                    break
                # FIX-TYPED-CHAT round 3 (2026-07-08): an EVI TEXT turn can take up to
                # ~9s to first audio (measured live) — a numeric grace kept losing the
                # race by a second and the clip's interrupt killed her reply. Now the
                # clip is held for as long as the typed reply is genuinely PENDING
                # (the flag clears the moment her reply lands); voice keeps 3s.
                _pend = getattr(self.state, "_typed_reply_pending", 0) or 0
                if time.time() - _pend < 13.0:
                    await asyncio.sleep(0.25)
                    continue
                if (not getattr(self.state, "_is_speaking", False)
                        and time.time() - _w0 > 3.0):
                    break
                await asyncio.sleep(0.25)
            if getattr(self.state, "_last_nova_at", 0) > _before:
                logger.info(f"[TURN] beat '{self.beat_name}' carried by her OWN reply (V2V — no clip needed)")
                return
        # No turn to ride → the clip is the mouth (instant, no generation, no dead air)
        logger.info(f"[TURN] beat '{self.beat_name}' — silent kid, clip covers it")
        await self.ask(fallback_line)

    def offer(self, kind, val):
        """ALL inputs enter here (typed / stt / detection / button).
        Resolved against the CURRENT beat only. Returns True if consumed.
        EAGER ANSWERS (2026-07-07 robot run): a kid who answers WHILE the ask is
        still playing is eager, not out of turn — buffer it; listen() replays it
        the moment the window opens."""
        if not self.listening or self._matcher is None:
            # 2026-07-08 live log 09:26: a REAL shoulder move landed milliseconds
            # after the window closed and was dropped → wrongly scored a miss.
            # A detection is a fact about the body — buffer it like typed/stt;
            # the retry window replays it and the kid gets their hit.
            if kind in ("typed", "stt", "detection"):
                self._eager = (kind, val, time.time())
                logger.info(f"[TURN] eager input '{kind}' buffered (beat={self.beat_name}) — replayed when the window opens")
            else:
                logger.info(f"[TURN] input '{kind}' outside listen window (beat={self.beat_name}) → dropped")
            return False
        try:
            r = self._matcher(kind, val)
        except Exception:
            r = None
        if r is None:
            # SIDE-CAPTURE: a name given in the wrong beat is still their name —
            # capture silently (context only), the beat continues undisturbed.
            if kind in ("typed", "stt") and not self.state.ctx.name:
                _nm = _extract_name(str(val))
                if _nm:
                    self.state.ctx.name = _nm
                    try:
                        memory.store.update(self.state.kid_id, name=_nm)
                    except Exception:
                        pass
                    logger.info(f"[TURN] side-captured name '{_nm}' (beat '{self.beat_name}' continues)")
            logger.info(f"[TURN] input '{kind}':'{str(val)[:30]}' does not resolve beat '{self.beat_name}'")
            return False
        self._result = (kind, r)
        self._last_resolution = (kind, time.time())
        self.listening = False
        self._got.set()
        logger.info(f"[TURN] beat '{self.beat_name}' RESOLVED by {kind}: {str(r)[:40]}")
        return True

    async def listen(self, matcher, timeout):
        """Open the listen window — inputs accepted only NOW (spec §2)."""
        self._matcher = matcher
        self._result = None
        self._got = asyncio.Event()
        self.listening = True
        logger.info(f"[TURN] listening ({self.beat_name}, {timeout}s window)")
        eager = getattr(self, "_eager", None)
        self._eager = None
        if eager is not None and time.time() - eager[2] < 10.0:
            logger.info(f"[TURN] replaying eager {eager[0]} input into '{self.beat_name}'")
            self.offer(eager[0], eager[1])
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.state.game_done.is_set() or not self.state.active:
                break
            if self.state.ctx.phase not in ("intro", "recognition"):
                logger.info(f"[TURN] phase left intro during '{self.beat_name}' → external resolution")
                self.listening = False
                return ("external", "phase")
            try:
                await asyncio.wait_for(self._got.wait(), timeout=0.25)
                break
            except asyncio.TimeoutError:
                continue
        self.listening = False
        self._matcher = None
        if self._result is not None:
            return self._result
        logger.info(f"[TURN] beat '{self.beat_name}' timed out ({timeout}s) → timeout action")
        return (None, None)


def _turn_wants_start(t):
    return _wants_to_start(str(t))


async def run_intro_turns(session: AgentSession, state: NovaSessionState,
                          agent: "NovaAgent", room: rtc.Room):
    """The intro as BEATS (FIX-TURN-OWNER). Replaces the old scattered flow."""
    eng = TurnEngine(session, state, room)
    state._turn_engine = eng

    # ── BEAT: greet — the reveal handler plays the ask (clip for new kids, EVI for
    # returning); we wait until it has been HEARD, then open listening.
    eng.new_beat("greet")
    _t0 = time.time()
    while getattr(state, "_last_nova_at", 0) == 0 and time.time() - _t0 < 45.0             and state.active and not state.game_done.is_set():
        await asyncio.sleep(0.3)
    await asyncio.sleep(0.4)

    # ── BEAT: name + mirror (COMMERCIAL-INTRO C.2/C.3 — pure V2V, air rules) ──
    # Kid answers → EVI heard the real audio and responds HERSELF (mirror).
    # Silence → ONE natural retry (~7s), ONE more, gentler (~8s), then "friend"
    # and she moves ON. The producer only LISTENS to advance (A.4) and whispers.
    def m_name(kind, val):
        if kind in ("typed", "stt"):
            nm = _extract_name(str(val))
            return nm if nm else None
        return None
    name = None
    kind = None
    if not state.ctx.name:
        kind, r = await eng.listen(m_name, 7.0)
        if kind in ("typed", "stt"):
            name = r
        if not name and kind is None:
            eng.new_beat("name_retry")
            await eng.cancel_stale()
            await eng.brief("gently ask their name ONE more time, one short warm line",
                            "what's your name, friend?")
            kind, r = await eng.listen(m_name, 8.0)
            if kind in ("typed", "stt"):
                name = r
        if not name and kind is None:
            eng.new_beat("name_retry2")
            await eng.cancel_stale()
            await eng.brief("one LAST, even gentler ask for their name — tiny and warm",
                            "it's okay — just your name, nice and loud!")
            kind, r = await eng.listen(m_name, 8.0)
            if kind in ("typed", "stt"):
                name = r
        if name:
            state.ctx.name = name
            try:
                memory.store.update(state.kid_id, name=name)
            except Exception:
                pass
            logger.info(f"[TURN] name captured: {name}")
            # A.4: producer LISTENS and silently arms the next stage — never intercepts
            _whisper(state, "MEMORY", f"their name is {name} — got it, the name talk is done forever")
        else:
            _whisper(state, "MEMORY", "no name given — call them 'friend', warm as ever, and NEVER re-ask")
    else:
        name = state.ctx.name

    # ── BEAT: move invite ──
    eng.new_beat("move_invite")
    await eng.cancel_stale()
    await eng.brief("invite them to try one little move together — ONE excited line ending "
                    "with the question 'ready to make a move?'", "ready to make a move?")
    def m_yes(kind2, val2):
        if kind2 in ("typed", "stt") and _turn_wants_start(val2):
            return "yes"
        return None
    await eng.listen(m_yes, 8.0)   # yes → straight in; timeout → she leads anyway

    # ── BEATS: the 2-CUE LIGHT CHALLENGE (COMMERCIAL-INTRO C.4 — a CHAIN):
    # cue → hit → INSTANT next cue ("it JUMPED to your HAND!") → hit → praise →
    # one live V2V beat (whispered details, her own words). Misses: one warm
    # retry → warm move-on. Zero fail-feel, ≤0.5s from detection to clip.
    hit = False
    chain_full = True
    state._challenge_ran = True   # legacy paths stay quiet
    for mv in INTRO_CHALLENGE:
        if state.ctx.phase not in ("intro", "recognition") or state.game_done.is_set():
            chain_full = False
            break
        eng.new_beat("challenge_" + mv["action"])
        await eng.cancel_stale()
        state._challenge_active = mv["action"]
        _det_t = getattr(eng, "_last_resolution", None)
        if hit and _det_t is not None:   # E: detection → next clip budget 500ms
            _lat = (time.time() - _det_t[1]) * 1000
            logger.info(f"[LAT-CLIP] detection → jump cue {int(_lat)}ms (budget 500ms)"
                        + ("" if _lat <= 500 else " — OVER BUDGET"))
        await eng.ask(mv["cue"])
        try:   # arm at ask END (spec §4) — a WOW can never fire mid-ask
            await room.local_participant.publish_data(
                json.dumps({"kind": "cue-part", "part": mv["action"], "joint": mv["joint"]}).encode("utf-8"),
                reliable=True)
            state._last_cue_part = (mv["action"], time.time())
            logger.info(f"[TURN] ask ended → light + detection ARMED on {mv['joint']}")
        except Exception:
            pass
        want = mv["action"]
        def m_move(kind3, val3, _want=want):
            return "hit" if (kind3 == "detection" and str(val3).lower() == _want) else None
        _k, r3 = await eng.listen(m_move, 7.0)
        if r3 != "hit":
            await eng.ask(mv["filler"])
            try:   # re-arm the browser's one-report-per-cue guard — without a fresh
                   # cue-part a second real move can never be reported (09:26 log)
                await room.local_participant.publish_data(
                    json.dumps({"kind": "cue-part", "part": mv["action"], "joint": mv["joint"]}).encode("utf-8"),
                    reliable=True)
                logger.info(f"[TURN] retry → light + detection RE-ARMED on {mv['joint']}")
            except Exception:
                pass
            _k, r3 = await eng.listen(m_move, 7.0)
        state._challenge_active = None
        if r3 == "hit":
            hit = True
            if mv["wow"]:   # chain end — the praise clip, then the live V2V beat
                await eng.ask(mv["wow"])
        else:
            chain_full = False
            break   # miss → warm move-on close (never fail-feel, never blame)
    seen = getattr(state, "_intro_moves_reacted", None) or set()
    seen.add("scripted")
    state._intro_moves_reacted = seen
    # C.4.d — the live soul: whisper the details; her natural voice reacts in her
    # own words the moment the kid gives her a turn. Detection shaky → no commentary.
    if hit:
        _whisper(state, "VISION",
                 ("they nailed the WHOLE light chain — the shoulder nudge AND the hand move, clean. "
                  if chain_full else "they nailed the shoulder nudge. ")
                 + "when they next speak, react to THAT in your own words — specific, delighted")
    else:
        _whisper(state, "VISION", "the light challenge ended without a clean detection — "
                 "NEVER mention it or blame them; pure energy, straight to the dance")

    # ── BEAT: play invite — she offers, then GENUINELY WAITS (never ask-and-do) ──
    if state.ctx.phase in ("intro", "recognition") and not state.game_done.is_set():
        eng.new_beat("play_invite")
        await eng.cancel_stale()
        state._dance_invited = True
        await eng.ask(_CHALLENGE_CLOSE_WIN if hit else _CHALLENGE_CLOSE_MISS)
        kind4, _r4 = await eng.listen(m_yes, 8.0)
        if state.ctx.phase in ("intro", "recognition") and not state.game_done.is_set():
            if kind4 is None:
                eng.new_beat("her_lead")
                await eng.cancel_stale()
                await eng.brief("they went quiet — take their hand warmly: one line saying "
                                "you'll pick a game together",
                                "come — I'll show you! let's pick a game together!")
            try:
                await room.local_participant.publish_data(
                    json.dumps({"kind": "go-picker"}).encode("utf-8"), reliable=True)
                logger.info("[TURN] picker opened (" + ("her lead after silence" if kind4 is None else "kid resolved") + ")")
            except Exception as e:
                logger.warning(f"[TURN] go-picker failed: {e}")
    state._turn_engine_done = True
    logger.info("[TURN] intro turns complete — engine idles; game phases are packet-driven")


# ════════════════════════════════════════════════════════════════════
# TALK SCORE ENGINE (2026-07-05, NOVA-SONG-TALK-SCORES.md)
# The browser announces song_start {song, sec} and 1s music_ticks; the
# worker plays the song's TALK SCORE off that clock — every line fired
# at (t_land - LEAD) so the EVI TTS lands ON the beat. Hard rules:
# one line per 2.5s (priority-drop), zero voice inside silence windows.
# ════════════════════════════════════════════════════════════════════
# ── NOVA V2V (NOVA-V2V-STRUCTURE.md, stage 0): the master flag. =1 switches the
# new closed-voice-loop path stage by stage; =0 (default) is today's build,
# untouched. Every V2V change MUST hide behind this flag until V2V-GOLD.
def _v2v_on() -> bool:
    return os.getenv("NOVA_V2V", "0") == "1"


def _gemini_on() -> bool:
    # GEMINI LIVE FALLBACK (2026-07-08): USE_GEMINI=1 swaps the voice to Gemini
    # Live (Hume credits ran out mid-test, user call: "use it, it's ok for now").
    # Takes precedence over EVI when set; flip back to Hume by unsetting it.
    return os.getenv("USE_GEMINI", "").lower() in ("1", "true", "yes", "on")


def _lemon_key() -> str | None:
    # STAGE 1 (2026-07-09): the key landed on the Render worker named
    # "lemonsilce" (typo). Accept every spelling seen so the env never needs
    # touching again; the canonical name still wins when present.
    return (os.getenv("LEMONSLICE_API_KEY") or os.getenv("LEMONSILCE_API_KEY")
            or os.getenv("lemonsilce") or os.getenv("LEMONSILCE")
            or os.getenv("lemonslice") or None)


_TALK_LEAD = float(os.getenv("NOVA_TALK_LEAD_SEC", "2.2"))   # measured: warm EVI delivery 2.2-3.4s
_TALK_CAP_SEC = 2.5


def _talk_now_sec(state) -> float:
    t0 = getattr(state, "_talk_t0", None)
    return (time.time() - t0) if t0 else 0.0


async def _talk_say_async(session: AgentSession, state: NovaSessionState, line: str, tag: str):
    """Non-blocking speech for the score: _nova_say awaits the WHOLE EVI generation
    (25s stalls seen live 08:30) — the scheduler must never wait on it, or beats slip.
    _say_inflight keeps lines from piling into EVI."""
    state._say_inflight = time.time()   # timestamp, not bool: cold gens ran 25s and froze 5 beats
    try:
        await _nova_say(session, line)
    finally:
        state._say_inflight = None
        state._last_nova_at = time.time()
        logger.info(f"[{tag}] line delivered → '{line}'")


async def _run_talk_score(session: AgentSession, state: NovaSessionState, song_id: str):
    score = personality.TALK_SCORES.get(song_id)
    if not score:
        logger.info(f"[TALK-SCORE] no score for '{song_id}' — old router stays")
        return
    if getattr(state, "_talk_score_song", None) == song_id and getattr(state, "_talk_score_active", False):
        return   # already running for this song
    state._talk_score_song = song_id
    state._talk_score_active = True
    state._talk_used = getattr(state, "_talk_used", {})
    state._echo_n = 0
    cap = float(score.get("min_gap", _TALK_CAP_SEC))   # wave rides tighter than 2.5s by design
    lead = 0.35 if (_CLIP_INDEX and os.getenv("NOVA_CLIPS", "1") == "1") else _TALK_LEAD
    logger.info(f"[TALK-SCORE] armed '{song_id}' — {len(score['beats'])} beats, lead {lead}s, gap {cap}s")
    try:
        for t_land, ref in score["beats"]:
            # wait for (t_land - LEAD) on the live music clock (re-synced by music_ticks)
            while True:
                if state.ctx.phase != "dance" or state.game_done.is_set() or not state.active:
                    logger.info(f"[TALK-SCORE] '{song_id}' stopped at beat {t_land}s (phase left dance)")
                    return
                wait = (t_land - lead) - _talk_now_sec(state)
                if wait <= 0:
                    break
                await asyncio.sleep(min(wait, 0.25))
            now_sec = _talk_now_sec(state)
            if now_sec - t_land > 3.0:
                logger.info(f"[TALK-SCORE] beat {t_land}s skipped (clock already at {now_sec:.1f}s)")
                continue
            if personality.talk_in_silence(song_id, t_land) and personality.talk_in_silence(song_id, now_sec):
                logger.info(f"[TALK-SCORE] beat {t_land}s inside a silence window — dropped")
                continue
            _fly = getattr(state, "_say_inflight", None)
            if _fly and time.time() - _fly < 6.0:
                # genuinely mid-flight → priority-drop. >6s = EVI cold-gen stall (25s seen
                # live, froze 5 beats): queue the next beat anyway — EVI plays in order.
                logger.info(f"[TALK-SCORE] beat {t_land}s dropped (a line is still speaking)")
                continue
            if time.time() - getattr(state, "_last_nova_at", 0) < cap:
                logger.info(f"[TALK-SCORE] beat {t_land}s dropped ({cap}s gap cap)")
                continue
            line = personality.talk_pool_pick(ref, state._talk_used)
            asyncio.create_task(_talk_say_async(session, state, line, "TALK-SCORE"))
            logger.info(f"[TALK-SCORE] {song_id} @{now_sec:.1f}s (land {t_land}s) → firing '{line}'")
    finally:
        state._talk_score_active = False
        logger.info(f"[TALK-SCORE] '{song_id}' score complete")


async def _talk_echo(session: AgentSession, state: NovaSessionState, ev_name: str):
    """Per-song HIT echo — the kid really moved; echo on every Nth hit, from the
    song's pool, never inside a silence window, never over the 2.5s cap."""
    song = getattr(state, "_talk_score_song", None)
    sc = personality.TALK_SCORES.get(song or "")
    pol = sc.get("echo") if sc else None
    if not pol:
        return
    sec = _talk_now_sec(state)
    if personality.talk_in_silence(song, sec):
        return
    state._echo_n = getattr(state, "_echo_n", 0) + 1
    every = int(pol.get("every", 2))
    if sec >= float(pol.get("fade_after", 1e9)):
        every = int(pol.get("fade_every", every))   # teacher fades as competence grows
    if state._echo_n % every:
        return
    _fly = getattr(state, "_say_inflight", None)
    if _fly and time.time() - _fly < 6.0:
        return
    if time.time() - getattr(state, "_last_nova_at", 0) < _TALK_CAP_SEC:
        return
    cur = (getattr(state.ctx, "current_move", "") or "").lower()
    pool = pol.get("clap_pool") if ("clap" in cur and pol.get("clap_pool")) else pol["pool"]
    line = personality.talk_pool_pick("@" + pool, getattr(state, "_talk_used", {}))
    logger.info(f"[TALK-ECHO] {ev_name} #{state._echo_n} @{sec:.1f}s → firing '{line}'")
    asyncio.create_task(_talk_say_async(session, state, line, "TALK-ECHO"))


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


async def _react_to_event(session: AgentSession, state: NovaSessionState,
                          agent: "NovaAgent", event: Optional[dict] = None):
    """Game event happened — pick tier (phrase bank vs LLM) for speed + cost.
    PHASE 3: when the per-song GameVoiceGate is armed, IT is the router
    (speak-gate → specialness → premade bank → live LLM). July path below
    stays as the NOVA_P3=0 fallback."""
    gate = getattr(state, "game_gate", None)
    if gate is not None and state.ctx.phase == "dance":
        ev = event or {"event": state.ctx.last_event or "hit", "streak": state.ctx.streak}
        return await _game_voice_event(session, state, agent, ev, gate)

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


# ────────────────────────────────────────────────────────────────────────
# PHASE 3 — the in-game voice path (voice-only, during the song).
# The GameVoiceGate decided; here we SPEAK: premade = instant bank line,
# live = two-stage (LLM ≤1s, bank covers if late/unsafe). Every decision
# logged as [P3-ROUTER] so a session log reads as a full voice transcript
# of silent/premade/live choices.
# ────────────────────────────────────────────────────────────────────────
async def _say_game_line(session: AgentSession, state: NovaSessionState, line: str) -> bool:
    """Speak one in-game line; on failure flip the gate to voice-down so the
    song continues on lights alone (edge 3). Success flips it back — she
    rejoins on reconnect without commenting on the gap."""
    if not line:
        return False
    gate = getattr(state, "game_gate", None)
    try:
        await state.pace.acquire()
        ok = await _nova_say(session, line)
    except Exception as e:
        logger.warning(f"[P3-ROUTER] say crashed (game continues on lights): {e}")
        ok = False
    if gate is not None:
        if ok and not gate.voice_ok:
            logger.info("[P3-ROUTER] voice back → rejoining silently (no comment on the gap)")
        gate.voice_ok = ok
    return ok


async def _game_voice_event(session: AgentSession, state: NovaSessionState,
                            agent: "NovaAgent", event: dict, gate) -> None:
    """One game event → one router decision → at most one spoken beat."""
    now = time.time()
    # Nova mid-line → drop, never stack (the gate also holds the cooldown).
    if state.pace._is_speaking and event.get("event") not in ("music_tick", "move_cue"):
        logger.info(f"[P3-ROUTER] ev={event.get('event')} → silent reason=nova_mid_line")
        return
    try:
        d = gate.decide(event, now)
    except Exception as e:
        logger.error(f"[P3-ROUTER] decide crashed → silent (game never depends on voice): {e}")
        return
    # keep the live context fresh for any LLM path
    state.ctx.kid_read = gate.kid_read(now)
    logger.info(f"[P3-ROUTER] t={d['t']}s ev={d['event']} → {d['action']}"
                f" key={d.get('key')} reason={d.get('reason')}"
                f" special={d.get('specialness')} kid_read={state.ctx.kid_read}")
    if d["action"] == "silent":
        return

    part = personality.move_friendly(event.get("part") or event.get("action")
                                     or state.ctx.current_move)
    side = event.get("side") or event.get("dir")

    if d["action"] == "premade":
        line = personality.pick_game_line(d["key"], part=part, side=side,
                                          name=state.ctx.name)
        await _say_game_line(session, state, line)
        logger.info(f"[P3-ROUTER] spoke premade[{d['key']}] → '{line}'")
        return

    # live path — two-stage: cached filler covers the call when available
    # (<50ms perceived), the LLM line lands ≤1s or the bank takes over.
    try:
        fp = getattr(state, "filler", None)
        cover = fp.claim(None, state.pace._is_speaking) if (fp and fp.enabled) else None
        if cover:
            asyncio.create_task(fp.fire(session, cover))
            logger.info(f"[P3-ROUTER] live cover=filler'{cover}'")
    except Exception:
        pass
    sysmsg, usermsg = personality.live_react_prompt(
        d["key"], {**event, "part": part}, state.ctx.name)
    line, source = await personality.speak_live_or_bank(
        lambda: _llm_line(sysmsg, usermsg, max_tokens=30),
        d["fallback_key"],
        timeout=float(os.getenv("NOVA_P3_LIVE_TIMEOUT", "1.0")),
        detection_ok=gate.detection_ok,
        part=part, side=side, name=state.ctx.name)
    await _say_game_line(session, state, line)
    logger.info(f"[P3-ROUTER] spoke live[{d['key']}] source={source} → '{line}'")


# PHASE 3: cancel the pipeline's automatic chatbot reply during the song —
# the gate decides how kid speech is handled (question → one line; story →
# tiny sound + continuity after the song). StopResponse is the official
# LiveKit hook for this; if this plugin version lacks it we fall back to
# letting the (already short, dance-persona) auto-reply through.
try:
    from livekit.agents.llm import StopResponse as _StopResponse
except Exception:
    try:
        from livekit.agents import StopResponse as _StopResponse
    except Exception:
        _StopResponse = None
        logger.warning("[P3-ROUTER] StopResponse unavailable — mid-song kid speech "
                       "falls through to the short dance-persona auto-reply")


async def _handle_dance_mic_text(session: AgentSession, state: NovaSessionState,
                                 agent: "NovaAgent", text: str) -> None:
    """Kid spoke mid-song. Route through the gate: direct question → ONE quick
    line, straight back to the game; chat/story → tiny sound now ('mm!') and
    Nova brings it up AFTER the song. Never full-stops the game."""
    gate = getattr(state, "game_gate", None)
    if gate is None:
        return
    await _game_voice_event(session, state, agent,
                            {"event": "mic_text", "text": text}, gate)


# COMMERCIAL-INTRO Part D (2026-07-08): the transition bridge. The producer
# WHISPERS the load state so she's aware of the machinery (never names it);
# her V2V rides any kid turn. When the kid is silent (a silent button tap has
# no turn), instant CLIPS cover the bridge beats — never a live generation.
_BRIDGE_CLIPS = {
    # song key 'joined' is kid-named "Up Groove" (legacy id) — spec Part B pitch
    ("hype", "hello"):  "Hello Hello?! YES — the song literally tells you what to do!",
    ("hype", "joined"): "Up Groove?! YES — we wake your body up, part by part!",
    ("hype", "freeze"): "Freeze Dance?! YES — when the music stops, you FREEZE!",
    ("hype", "wave"):   "Wave?! YES — the magic light travels up your whole body!",
    ("tip", None):      "when the light glows — that's me!",
    ("framing", None):  "step back so I can see ALL of you!",
    # 'framed' is a whisper-only beat: the GO-BEAT fires once, at phase:dance
    ("fail", None):     "hmm, that one's being shy! pick another!",
}


def _bridge_clip_line(beat, song):
    line = _BRIDGE_CLIPS.get((beat, (song or "").strip() or None))
    if line is None:
        line = _BRIDGE_CLIPS.get((beat, None))
    return line


async def _speak_bridge(session: AgentSession, state: NovaSessionState, agent: "NovaAgent",
                        beat: Optional[str], song: Optional[str]):
    """PHASE 2 TRANSITION — ONE bridge beat the browser asked for. The browser
    owns the budget + ready-gate. Part D: whisper the stage, clip the voice."""
    _whisper(state, "STAGE",
             f"the game '{song or '?'}' — bridge beat '{beat}': it's getting the music ready; "
             "stay hyped, no tech words, max one short line if they speak")
    line = _bridge_clip_line(beat, song)
    if not line:
        logger.info(f"[bridge] {beat}/{song} → (whisper only, no voice)")
        return
    await state.pace.acquire()
    await _nova_say(session, line)   # exact manifest text → instant clip, no generation
    logger.info(f"[bridge] {beat}/{song} → clip '{line}'")


async def _speak_dance_intro(session: AgentSession, state: NovaSessionState, agent: "NovaAgent"):
    """Phase transitioned to dance — the GO-BEAT (Part D.3), a timed game moment:
    one instant clip; the browser starts the song the moment it begins."""
    await state.pace.acquire()
    await _nova_say(session, "here we GO!")


async def _speak_goodbye(session: AgentSession, state: NovaSessionState, agent: "NovaAgent"):
    """THE ENDING (2026-07-05, NOVA-ENDING.md): RETURN → ONE REAL CALLBACK →
    PLANT TOMORROW (the comeback deposit) → GOODBYE. Length scales with the game
    (wave ≤3 lines, full songs ≤4). Edge cases: quit mid-song = 2 warm lines, no
    ceremony; kid already gone = one soft line; zero hits = tech takes the blame.
    Finishes with a goodbye-done packet so the browser shows stars/feedback ONLY
    after her last word."""
    if getattr(state, "_goodbye_ran", False):
        return
    state._goodbye_ran = True
    await state.pace.acquire()
    memory.store.increment_sessions(state.kid_id)

    song = getattr(state, "_talk_score_song", None) or "hello"
    dur = personality.SONG_DUR.get(song, 90.0)
    sec = float(getattr(state.ctx, "music_sec", 0) or 0)
    completed = sec >= 0.8 * dur
    hit_actions = getattr(state, "_hit_actions", set())
    zero_hits = (state.ctx.hits or 0) == 0
    topic = getattr(state.ctx, "deferred_topic", None)
    last_key = (state.ctx.shared_facts or {}).get("deposit_key")

    # PLANT TOMORROW — the deposit is saved FIRST (play-again keeps it too),
    # and the NEXT session's intro opens with deposit_intro.
    dep_line, dep_key, dep_intro = personality.pick_deposit(song, topic, completed, last_key)
    try:
        memory.store.add_shared_fact(state.kid_id, "deposit_key", dep_key)
        memory.store.add_shared_fact(state.kid_id, "deposit_intro", dep_intro)
    except Exception as e:
        logger.warning(f"[ENDING] deposit save failed: {e}")

    if getattr(state, "_goodbye_skip", False):          # play-again: pure delight, no ceremony
        lines = ["AGAIN?! okay okay—"]
    elif getattr(state, "_kid_away", False):            # edge 2: never a speech to nobody
        lines = ["bye friend — I'll be here!"]
    elif not completed:                                 # edge 1: quit mid-song, no ceremony
        lines = ["hey — that was FUN. come finish it with me tomorrow?"]
    else:
        quick = bool((personality.GOODBYE_SCORES.get(song) or {}).get("quick"))
        cb = (personality.GOODBYE_TECHBLAME if zero_hits else
              (personality.pick_goodbye_callback(song, hit_actions, state.ctx.hits or 0)
               or personality.GOODBYE_BRAVERY))
        if quick:   # wave: a 28s game can't earn a ceremony — 3 lines, fast-bright
            lines = [cb, dep_line, "see you tomorrow — I'll be here!"]
        else:       # full songs: the 4 beats
            lines = ["okay okay — come here!", cb, dep_line, "same time tomorrow? …I'll be here!"]

    t0 = time.time()
    for ln in lines:
        await _nova_say(session, ln)
        await asyncio.sleep(0.35)
    logger.info(f"[ENDING] {song} completed={completed} zero_hits={zero_hits} "
                f"lines={len(lines)} deposit='{dep_key}' spoke_in={time.time()-t0:.1f}s")
    # stars/feedback appear only AFTER her last word
    try:
        room = getattr(state, "room", None)
        if room:
            await room.local_participant.publish_data(
                json.dumps({"kind": "goodbye-done"}).encode("utf-8"), reliable=True)
    except Exception:
        pass


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
        await _nova_say(session, text)   # EVI-safe (raw say() is silent under Hume)
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


import re as _re
_START_PHRASE = _re.compile(r"\blet'?s\s+(start|dance|play|go)\b|\bstart\s+the\s+game\b|\bi'?m\s+ready\b", _re.I)
_START_WORD = _re.compile(r"\b(start|dance|play|go|ready|yes|yalla|ok(?:ay)?|begin)\b", _re.I)
_NEGATION = _re.compile(r"\b(no|not|don'?t|stop|wait|later)\b", _re.I)


def _wants_to_start(text: str) -> bool:
    """True only for a REAL let's-go: explicit phrase anywhere, or a short (≤3 word)
    positive burst like 'yes!' / 'dance!' / 'ready'. Never on negations, never on
    keywords buried in chatter ('I like to play soccer')."""
    t = (text or "").strip()
    if not t or _NEGATION.search(t):
        return False
    if _START_PHRASE.search(t):
        return True
    return len(t.split()) <= 3 and bool(_START_WORD.search(t))


async def _silence_driver(state: NovaSessionState, session: AgentSession):
    """INTRO-4FIXES #4: she LEADS on silence. During recognition, if the dancer stays
    quiet ~7s after her last line, nudge her brain to apply the current beat's silence
    rule (retry once → fallback → ADVANCE). Max 5 nudges; stops when the phase moves on.
    Timestamps: state._last_nova_at / _last_kid_at are set by the speak/hear hooks."""
    nudges = 0
    if getattr(state, "_turn_engine", None) is not None:
        logger.info("[SILENCE-DRIVER] turn engine owns the intro — driver stands down")
        return
    logger.info(f"[SILENCE-DRIVER] armed (phase={state.ctx.phase}, active={state.active})")
    while state.active and nudges < 2:   # 2 attempts into silence MAX — then she waits quietly
        await asyncio.sleep(2.0)
        try:
            if state.ctx.phase not in ("intro", "recognition"):
                logger.info(f"[SILENCE-DRIVER] phase moved to {state.ctx.phase} → standing down")
                return
            if getattr(state, "_dance_invited", False):
                # invite is out — nudging now replays stale challenge beats
                # ("still waiting on that nudge…" AFTER the picker, live 08:39)
                logger.info("[SILENCE-DRIVER] dance invited → standing down for good")
                return
            now = time.time()
            ln, lk = getattr(state, "_last_nova_at", 0), getattr(state, "_last_kid_at", 0)
            if not ln:                      # she hasn't spoken yet — wait
                continue
            if now - ln < 10.0 or (lk and now - lk < 10.0):
                continue                    # someone spoke recently — conversation alive
            nudges += 1
            logger.info(f"[SILENCE-DRIVER] {int(now-ln)}s quiet → nudge #{nudges} (retry/fallback/advance)")
            model = getattr(state, "_evi_model", None)
            if model is not None:
                for sess in list(getattr(model, "_sessions", [])):
                    try:
                        sess._send({"type": "user_input",
                                    "text": "(silence — apply the silence rule for the current step: one shorter retry if you haven't retried, otherwise use the fallback and move to the NEXT step)"})
                        break
                    except Exception:
                        pass
            else:
                await session.generate_reply(
                    instructions="(the dancer is silent — retry the question once shorter, or use the fallback and advance)")
            state._last_nova_at = time.time()   # count the nudge as her turn (prevents machine-gunning)
        except Exception as e:
            logger.warning(f"[SILENCE-DRIVER] {e}")
    logger.info("[SILENCE-DRIVER] done (cap or phase change)")


async def _push_to_game(state: NovaSessionState, session: AgentSession, line: str):
    """Nova TAKES the kid to the game: speak one hype line + tell the browser to open
    the game picker. She is a dance coach, not a chatbot — this is her exit move."""
    try:
        await _nova_say(session, line)
    except Exception:
        pass
    try:
        room = getattr(state, "room", None)
        if room:
            await room.local_participant.publish_data(
                json.dumps({"kind": "go-picker"}).encode("utf-8"), reliable=True)
            logger.info("[GAME-PUSH] sent go-picker → browser opens the game picker")
    except Exception as e:
        logger.warning(f"[GAME-PUSH] failed: {e}")


# What SHE says drives the world: name a body part → the browser lights it on the kid's
# camera; call the dance → the game picker opens. Parsed from her live assistant lines.
_NOVA_PART = [
    (_re.compile(r"\bclap|hands?\s+together\b", _re.I), "hands"),
    (_re.compile(r"\b(right|left|both)?\s*hands?\s+(up|high|raise)|raise\s+(your\s+)?(right|left|both)?\s*hand\b", _re.I), "hands"),
    (_re.compile(r"\bshoulders?\b", _re.I), "shoulder"),
    (_re.compile(r"\bhead\b", _re.I), "head"),
]
_NOVA_GO = _re.compile(r"\blet'?s\s+(dance|play)\b|\bpick\s+(a|your)\s+game\b|\btime\s+to\s+move\b|\bpush\s+the\s+(big\s+)?button\b", _re.I)


def _scan_nova_line(state: NovaSessionState, txt: str):
    """Fire-and-forget: mirror Nova's words into browser actions (intro/recognition only)."""
    try:
        # NOTE: _run_nova sets phase "intro" — the old "recognition"-only check
        # meant the intro try-a-move light NEVER fired in a live session.
        # "play" included too: a premature 'yes' signal advances the worker phase
        # while the kid is still on the intro screen — her body-part cues must
        # keep lighting up there (browser gates by ITS OWN phase, so this is safe).
        if getattr(state.ctx, "phase", "") not in ("intro", "recognition", "play"):
            return
        # during the SCRIPTED challenge the worker owns cue/light/flow — her spoken
        # lines (which name the parts) must not re-cue or open the picker
        if getattr(state, "_challenge_active", None):
            return
        room = getattr(state, "room", None)
        if not room:
            return
        if _NOVA_GO.search(txt or ""):
            # TURN-OWNER §2: she NEVER performs the action she just offered — the
            # engine opens the picker on the kid's answer or its own timeout lead.
            state._dance_invited = True
            return
        for pat, part in _NOVA_PART:
            if pat.search(txt or ""):
                # her own REACTION line names the part too ("that SHOULDER popped") —
                # don't re-cue the same part in a burst, it re-arms the browser detector
                last = getattr(state, "_last_cue_part", None)
                if last and last[0] == part and time.time() - last[1] < 12.0:
                    return
                state._last_cue_part = (part, time.time())
                logger.info(f"[CUE-PART] Nova named '{part}' → lighting it on the kid")
                asyncio.create_task(room.local_participant.publish_data(
                    json.dumps({"kind": "cue-part", "part": part}).encode("utf-8"), reliable=True))
                return
    except Exception as e:
        logger.warning(f"[scan-nova-line] {e}")


async def _user_said(session: AgentSession, state: NovaSessionState, agent: "NovaAgent", text: str):
    """Kid spoke or typed. Inject as user input + extract knowledge + save to memory."""
    logger.info(f"[TYPE] kid typed → '{text[:80]}'")

    # Update live context so next prompt build has the kid's words +
    # knowledge.py can detect topic mentions (colors, animals, foods)
    state.ctx.last_kid_text = text
    # FIX-TYPED-CHAT: typed = spoken, one pipeline — typing counts as the kid's
    # voice everywhere (brief's turn-gap, dead-air court, filler guards)...
    state.last_kid_speech_at = time.time()
    state._lat_kid_final_at = time.time()   # ...and the reply-latency meter
    # ...and the STT-echo bubble speaks one visual language for both inputs
    try:
        _room = getattr(state, "room", None)
        if _room is not None:
            asyncio.create_task(_room.local_participant.publish_data(
                json.dumps({"kind": "stt-echo", "text": text[:120]}).encode("utf-8"),
                reliable=True))
    except Exception:
        pass

    # PHASE 3: mid-song typed text routes through the gate too (question →
    # one quick line, story → tiny sound + after-song continuity). Never a
    # full chatbot reply while they dance.
    if (getattr(state.ctx, "phase", "") == "dance"
            and getattr(state, "game_gate", None) is not None):
        await _handle_dance_mic_text(session, state, agent, text)
        return

    # ── TURN-OWNER: typed inputs go to the ONE engine, resolved against the
    # current beat only. Consumed → the beat advances silently; on NAME beats the
    # text still flows to EVI below (COMMERCIAL-INTRO C.2 — a typed name is a real
    # kid turn and SHE does the mirror, not a script). Other consumed beats stop
    # here: their clips are the mouth (one beat = one mouth).
    _eng = getattr(state, "_turn_engine", None)
    if _eng is not None and _eng.offer("typed", text):
        try:
            memory.store.add_message(state.kid_id, "user", text)
        except Exception:
            pass
        if (getattr(_eng, "beat_name", "") or "").startswith(("greet", "name")) and _v2v_on():
            logger.info("[chat] typed name consumed by beat → still forwarded to EVI (her mirror)")
        else:
            return

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
    # TYPED-REPLY FIX (2026-07-06 live): EVI SWALLOWS a text user_input that lands while
    # SHE is speaking — [HEAR] confirmed the text arrived, but no assistant turn ever
    # started (typed chat felt dead). FIX-TYPED-CHAT (2026-07-08): a playing CLIP is her
    # audio too — a user_input sent mid-clip gets its reply suppressed by the one-mouth
    # guard (clip-owned beat) and dies. Barge-in rule: queue until her line-end (cap 8s),
    # and if no reply begins within 5s, resend the text once.
    for _ in range(32):
        if (not getattr(state, "_is_speaking", False)
                and not getattr(state, "_clip_playing", False)):
            break
        await asyncio.sleep(0.25)
    _before = getattr(state, "_last_nova_at", 0)
    state._order_n = getattr(state, "_order_n", 0) + 1
    state._typed_reply_pending = time.time()   # brief() holds clips while this is fresh
    logger.info(f"[ORDER] #{state._order_n} directed-reply (typed conversation): '{text[:40]}'")
    logger.info("[BRAIN] generating reply to kid input...")
    # FIX-TYPED-CHAT (2026-07-08): send the text STRAIGHT to EVI as a user turn.
    # The framework hop (AgentSession.generate_reply(user_input=…)) silently
    # dropped it — wire diagnostics showed no user_input ever left the worker.
    _model = getattr(state, "_evi_model", None)
    try:
        if _model is not None and _v2v_on() and hasattr(_model, "send_user_text"):
            _model.send_user_text(text)
        else:
            await session.generate_reply(user_input=text)
        logger.info(f"[BRAIN] reply call returned for: '{text[:40]}'")
    except Exception as e:
        logger.exception(f"[BRAIN] generate_reply FAILED for kid input: {e}")

    async def _ensure_reply():
        # STAGE 2.2 (2026-07-09): ONE reply per turn, EVER. The 5s deadline kept
        # losing to the slow FIRST generation of a session (7-10s measured) — the
        # retry then produced a second answer back-to-back. Now: poll up to 12s
        # and CANCEL the retry the instant her reply's audio starts.
        for _ in range(48):
            await asyncio.sleep(0.25)
            if (getattr(state, "_last_nova_at", 0) > _before
                    or getattr(state, "_is_speaking", False)):
                return   # her reply started — retry cancelled
            if not state.active or state.game_done.is_set():
                return
        if getattr(state, "_last_nova_at", 0) <= _before:
            # barge-in rule holds for the retry too: never resend INTO her audio
            for _ in range(20):
                if (not getattr(state, "_is_speaking", False)
                        and not getattr(state, "_clip_playing", False)):
                    break
                await asyncio.sleep(0.25)
            if (getattr(state, "_last_nova_at", 0) > _before
                    or getattr(state, "_is_speaking", False)):
                return   # her reply landed while we waited — done
            logger.info(f"[BRAIN] typed text got NO reply in 12s → resending once: '{text[:40]}'")
            # VOICE-SILENCE DEBUG (2026-07-09): surface the voice engine's real
            # failure into the room — a probe/browser can read what Render logs say
            try:
                _room = getattr(state, "room", None)
                if _room is not None:
                    await _room.local_participant.publish_data(json.dumps({
                        "kind": "voice-error",
                        "err": getattr(_model, "last_error", None) or "(no exception recorded — reply never started)",
                        "text": text[:60],
                    }).encode("utf-8"), reliable=True)
            except Exception:
                pass
            try:
                if _model is not None and _v2v_on() and hasattr(_model, "send_user_text"):
                    _model.send_user_text(text)
                else:
                    await session.generate_reply(user_input=text)
            except Exception as e2:
                logger.warning(f"[BRAIN] typed-text resend failed: {e2}")
    asyncio.create_task(_ensure_reply())


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

        # COMMERCIAL-INTRO C.3 (the air rule): nobody in frame = QUIET waiting
        # state. Zero talking to the air — the presence handler owns that beat.
        if not getattr(state, "_kid_present", True):
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
            await _nova_say(session, line)   # EVI-safe (raw say() is silent under Hume)
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
    """Speak a scripted game line, serialized through the pace gate.
    HUME-ONLY (2026-07-02): routes through _nova_say — raw session.say() is silent
    under EVI (supports_say=False), which muted every in-game callout."""
    if not state.active or state.game_done.is_set():
        return
    try:
        await state.pace.acquire()
        await _nova_say(session, line)
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
    # words Nova herself says (mic echo / barge-in) — "Ready" got stored as a
    # name from "Ready to get those dancing shoes on?" (2026-07-04 session)
    "ready", "start", "lets", "dance", "dancing", "dancer", "sure", "fine",
    "please", "thanks", "thank", "sorry", "friend", "again", "come", "wow",
    "superstar", "champion", "awesome", "great", "fantastic",
}

def _extract_name(text: Optional[str]) -> Optional[str]:
    """Best-effort name pull. STRONG patterns ('my name is X') win even inside a
    question; otherwise only a clean short statement (not a question) counts."""
    if not text:
        return None
    import re
    t = text.strip()
    # Strong explicit pattern — trust it even if the sentence has a '?'
    m = re.search(r"(?:my name is|i'?m|i am|im|call me|name's|they call me)\s+([A-Za-z][A-Za-z\-']{1,20})", t, re.I)
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
    return bool(re.search(r"(?:my name is|call me|name's|they call me|i'?m |i am |im )", text, re.I))


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

    # DIRECT-GAME (?game= link): no intro flow at all — the browser auto-launches the
    # game; the worker rides along (talk score + ending are packet-driven).
    if getattr(state, "_direct_game", None):
        logger.info("[nova] DIRECT-GAME session → intro flow skipped entirely")
        while state.active and not state.game_done.is_set():
            await asyncio.sleep(1.0)
        await _end_game(session, state, agent, state.ctx.observed_visual)
        return

    # ── TURN ENGINE (FIX-TURN-OWNER 2026-07-07): the ONE conversation owner.
    # greet → name → move-invite → challenge (arm at ask END) → play-invite →
    # picker. Every beat: ask → heard → listen → resolve/timeout → advance.
    await run_intro_turns(session, state, agent, room)

    # engine idles; the game/goodbye phases are packet-driven from here
    while state.active and not state.game_done.is_set():
        await asyncio.sleep(1.0)

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
    # THE ENDING owns the goodbye (2026-07-06 live: a SECOND long adult goodbye —
    # with the PREVIOUS kid's name from memory — spoke after the real 4-beat close).
    # If the scripted ending already ran, this legacy exit line stays silent.
    if getattr(state, "_goodbye_ran", False):
        logger.info("[END] scripted ending already spoke — legacy exit line skipped")
        return
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
    voice_only = False
    direct_game = None
    # ROOT-CAUSE FIX (2026-07-05): ctx.room.metadata is EMPTY before ctx.connect() —
    # every session ever ran as anon-* (memory never got real kid ids) and the
    # voiceOnly/directGame flags were silently lost. The JOB carries the room info
    # (and the dispatch metadata) BEFORE connect — read all three sources.
    _meta_sources = []
    try:
        _meta_sources.append(("room", ctx.room.metadata))
    except Exception:
        pass
    try:
        _meta_sources.append(("job.room", getattr(getattr(ctx.job, "room", None), "metadata", None)))
    except Exception:
        pass
    try:
        _meta_sources.append(("job", getattr(ctx.job, "metadata", None)))
    except Exception:
        pass
    for _src, _raw in _meta_sources:
        if not _raw:
            continue
        try:
            meta = json.loads(_raw)
            kid_id = meta.get("kidId") or kid_id
            voice_only = voice_only or bool(meta.get("voiceOnly"))
            direct_game = direct_game or meta.get("directGame") or None
            logger.info(f"[nova-v207] metadata via {_src}: kidId={kid_id} voiceOnly={voice_only} directGame={direct_game}")
            break
        except Exception:
            continue

    state = NovaSessionState(kid_id=kid_id)
    if direct_game:
        # DIRECT-GAME (?game= link): the browser jumps straight into the game — NO intro
        # name-talk, NO scripted challenge (it talked over the game, live 2026-07-04).
        state._challenge_ran = True
        state._dance_invited = True
        state._direct_game = direct_game
        logger.info(f"[nova-v207] DIRECT-GAME session → intro challenge OFF, game '{direct_game}'")
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

    # V2V-ONLY (2026-07-08, user orders): Deepgram STT deleted — the ear is
    # Hume EVI or Gemini Live. If neither flag is set there is NO STT fallback,
    # so fail loud here instead of starting a session that can't hear.
    if not (_evi_on() or _gemini_on()):
        logger.error("[nova-v207] FATAL: no realtime voice — set USE_EVI=1 (+NOVA_FORCE_ELEVENLABS=0) or USE_GEMINI=1")
        raise RuntimeError("V2V-ONLY: USE_EVI=1 or USE_GEMINI=1 required (Deepgram STT deleted 2026-07-08)")

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

    # Build session pipeline.
    # NO STT: Deepgram deleted (2026-07-08). EVI hears the kid directly
    # (speech-to-speech) — the _evi_on() guard above makes EVI mandatory, and the
    # EVI branch below replaces this dict wholesale.
    session_kwargs = dict(
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

    # RECONNECT GRACE (2026-07-07 live run): a one-second network blip used to end
    # the session for good — the kid's page came right back, but state.active was
    # already False and (worse) the AgentSession had closed, so the rest of the run
    # was clips-only with "AgentSession isn't running" errors. A disconnect now
    # starts a 60s grace timer; only a kid who truly never returns stops the loops.
    # Room close (below) stays immediate — that IS final.
    _kid_gone = {"t": None}

    @ctx.room.on("participant_disconnected")
    def _on_participant_left(participant):
        ident = getattr(participant, "identity", "?")
        if str(ident).startswith(("runway-", "agent-", "nova")):
            return
        _kid_gone["t"] = time.time()
        logger.info(f"[nova-v207] kid {ident} disconnected → 60s reconnect grace")

        async def _grace_kill():
            await asyncio.sleep(60.0)
            if _kid_gone["t"] is not None and time.time() - _kid_gone["t"] >= 59.0:
                state.active = False
                logger.info("[nova-v207] kid never returned (60s grace) → stopping session loops")

        asyncio.create_task(_grace_kill())

    @ctx.room.on("participant_connected")
    def _on_participant_joined(participant):
        ident = getattr(participant, "identity", "?")
        if str(ident).startswith(("runway-", "agent-", "nova")):
            return
        if _kid_gone["t"] is not None:
            logger.info(f"[nova-v207] kid {ident} RECONNECTED → grace cancelled, session continues")
        _kid_gone["t"] = None

    @ctx.room.on("disconnected")
    def _on_room_disconnected(*_a):
        state.active = False
        logger.info("[nova-v207] room disconnected → stopping idle loop")

    # ── GEMINI LIVE (fallback voice, 2026-07-08) — USE_GEMINI=1 wins ─────
    _gemini_adapter = None
    if _gemini_on():
        try:
            from livekit.plugins import google as google_rt
            from gemini_voice import GeminiVoiceAdapter
            evi_prompt = None
            try:
                evi_prompt = personality.build_evi_system_prompt(state.ctx, direct_game=direct_game)
                logger.info(f"[nova-gemini] session prompt built ({len(evi_prompt)} chars)")
            except Exception as pe:
                logger.exception(f"[nova-gemini] prompt build FAILED → minimal prompt: {pe}")
                evi_prompt = "You are NOVA, a warm magical movement friend. Short playful replies, always lead toward a dance."
            _gemini_adapter = GeminiVoiceAdapter(system_prompt=evi_prompt)
            _gemini_adapter._ears_open = False   # door CLOSED until reveal (same law as EVI)
            _gemini_adapter._mouth_hold = True
            if direct_game:
                _gemini_adapter._greet_text_override = (
                    "(the dancer just arrived and the dance game is starting "
                    "right now — greet them with ONE short excited line, no questions)")
            state._evi_model = _gemini_adapter
            # ROOT CAUSE FIX (2026-07-09, "she doesn't reply to text"): the plugin
            # IGNORES generate_reply on any "3.1" live model (capabilities.
            # mutable_chat_context=False → instantly-failed future, swallowed by
            # AgentSession — replies "complete" in 0.03s with zero audio). Every
            # worker-directed line (typed chat, greeting, briefs, whispers) died
            # there. gemini-2.5-flash-native-audio is the plugin's own default and
            # fully supports directed replies (probe-verified with the worker key).
            _gemini_model = os.getenv("NOVA_GEMINI_MODEL",
                                      "gemini-2.5-flash-native-audio-preview-12-2025")
            session_kwargs = {
                "llm": google_rt.realtime.RealtimeModel(
                    model=_gemini_model,
                    voice=os.getenv("NOVA_GEMINI_VOICE", "Leda"),
                    instructions=evi_prompt,
                ),
                "allow_interruptions": True,
            }
            logger.info(f"[nova-gemini] USE_GEMINI=1 → voice = Gemini Live "
                        f"({_gemini_model}, "
                        f"voice {os.getenv('NOVA_GEMINI_VOICE', 'Leda')})")
        except Exception as e:
            logger.exception(f"[nova-gemini] init FAILED — refusing silent fallback: {e}")
            raise

    # ── HUME EVI 3 (speech-to-speech) — THE voice, no fallback ──────────
    # HUME-ONLY (user decision 2026-07-02): when EVI is on, Nova speaks Kora or she
    # doesn't start — NO silent switch to ElevenLabs (that's how the wrong voice shipped).
    # An init failure raises → the job dies loudly → the browser shows retry, and the
    # logs say exactly why (auth/credits/websocket).
    elif _evi_on():
        try:
            from evi_realtime import HumeEVIRealtimeModel
            # PHASE 1 (2026-07-03): the recognition brain-prompt is built PER SESSION
            # (kid name / returning / callbacks / tier) and sent as the EVI session
            # system_prompt — personality.py is her live brain again, per kid.
            # builder failure must NOT kill the session — Hume config v1 (coach prompt)
            # is the wired fallback. Hume-only still holds: voice is Kora either way.
            evi_prompt = None
            try:
                evi_prompt = personality.build_evi_system_prompt(state.ctx, direct_game=direct_game)
                logger.info(f"[nova-evi] session system_prompt built ({len(evi_prompt)} chars, returning={bool(state.ctx.name and state.ctx.sessions_before>=1)})")
            except Exception as pe:
                logger.exception(f"[nova-evi] prompt build FAILED → using Hume config prompt: {pe}")
            _evi_model = HumeEVIRealtimeModel(system_prompt=evi_prompt)   # keys/config from env
            # ONE-BEAT-ONE-MOUTH (2026-07-07 hara.txt): pre-reveal audio used to pile
            # up in EVI and come back as a live reply ON TOP of the greet clip. Ears
            # stay CLOSED until reveal-now, and the mouth is HELD until the first
            # worker-ordered speech or kid input after a clip.
            _evi_model._ears_open = False
            _evi_model._mouth_hold = True
            if _v2v_on():
                logger.info("[EARS] door CLOSED until reveal (no pre-reveal turns)")
            state._evi_model = _evi_model   # reveal-now handler fires the greeting through this
            if direct_game:
                # game link: one excited hello, no name question — the game starts in seconds
                _evi_model._greet_text_override = ("(the dancer just arrived and the dance game is starting "
                                                   "right now — greet them with ONE short excited line, no questions)")
            session_kwargs = {
                "llm": _evi_model,
                "allow_interruptions": True,
            }
            logger.info("[nova-evi] USE_EVI=1 → STT+LLM+TTS replaced with Hume EVI realtime (Kora)")
        except Exception as e:
            logger.exception(f"[nova-evi] EVI init FAILED — HUME-ONLY mode, refusing ElevenLabs fallback: {e}")
            raise

    session = AgentSession(**session_kwargs)
    logger.info("[nova-v207] step 1: AgentSession created")
    if _gemini_adapter is not None:
        _gemini_adapter.bind(session)   # ears/mouth control surface goes live

    # Wire the pre-cached filler system (instant reactions while the LLM cooks).
    # Guarded so a filler problem can never block the session from starting.
    state.session = session
    session._nova_state = state   # _nova_say reads room/state through the session (clip path)
    try:
        state.filler = FillerPlayer()
        logger.info(f"[filler] system ready (enabled={state.filler.enabled})")
    except Exception as e:
        logger.error(f"[filler] init failed, fillers disabled: {e}")
        state.filler = None

    # ─────────────────────────────────────────────────────────────
    # HEAVY LOGGING HOOKS — distinct log line at each pipeline stage.
    # Each step gets a tag so we can tell exactly WHERE a session breaks:
    #   [HEAR]  — kid's voice transcribed (Hume EVI's own ASR — Deepgram deleted 2026-07-08)
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
            # Log BOTH interim and final so we can see the ear (EVI ASR) is alive at all
            if is_final:
                state.bump("turns")
                logger.info(f"[HEAR] final ✓ kid voice → '{text[:80]}'")
                # Feed the move-game orchestrator: pulse "kid spoke" + classify
                # the utterance as a done/yes signal so the loop can react.
                state.last_kid_text = text
                state.last_kid_speech_at = time.time()
                state._lat_kid_final_at = time.time()   # E: reply-latency start mark
                # C.2 STT-ECHO (Lexi steal): the browser shows "Nova heard: …" so
                # truncations are visible instead of silently rolled past.
                try:
                    asyncio.create_task(ctx.room.local_participant.publish_data(
                        json.dumps({"kind": "stt-echo", "text": text[:120]}).encode("utf-8"),
                        reliable=True))
                except Exception:
                    pass
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
                # TURN-OWNER: voice inputs go to the engine (resolved against the
                # current beat only). EVI replies to voice naturally either way.
                _eng = getattr(state, "_turn_engine", None)
                if _eng is not None:
                    _eng.offer("stt", text)
                sig = personality.detect_signal(text)
                if (sig == "yes" and state.ctx.phase in ("intro", "recognition")
                        and not getattr(state, "_turn_engine_done", False)):
                    sig = None   # the engine owns intro yes-handling
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

    # Direct error visibility — if the session's audio path is rejecting input we'll see it
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
            state._is_speaking = speaking   # mute-guard checks this (text lands later than audio)
            state.pace.mark_speaking(speaking)
            # STAGE 2.1 (2026-07-09): SHE MUST NEVER HEAR HERSELF. The clip path
            # already closes the Gemini door; her LIVE voice did not — echo of her
            # own line came back through the kid's mic and she answered herself.
            # Door CLOSED while she speaks, restored to the phase's base state
            # after (same rule as the post-clip restore). Gemini path only — Hume
            # EVI is one S2S socket and handles its own echo.
            if _gemini_on():
                _vm = getattr(state, "_evi_model", None)
                if _vm is not None and _v2v_on():
                    if speaking:
                        _vm._ears_open = False
                    elif not getattr(_vm, "_clip_playing", False):
                        _vm._ears_open = (getattr(state.ctx, "phase", "intro")
                                          in ("intro", "recognition", "goodbye"))
            # COMMERCIAL-INTRO Part E: reply latency = kid's final transcript → her
            # audio starts. Target ≤1.5s, hard ceiling 3s (ERROR above it).
            if speaking:
                _mark = getattr(state, "_lat_kid_final_at", None)
                if _mark:
                    state._lat_kid_final_at = None
                    _dt = time.time() - _mark
                    if _dt <= 3.0:
                        logger.info(f"[LAT] reply {_dt:.2f}s (target 1.5s)")
                    else:
                        logger.error(f"[LAT] reply {_dt:.2f}s — OVER the 3s ceiling")
            # Part D.5 state badge: Watching / Listening / Talking / Thinking —
            # silence must always read as intentional on screen.
            _badge = {"speaking": "talking", "listening": "listening",
                      "thinking": "thinking"}.get(str(new_state or ""), None)
            if not getattr(state, "_kid_present", True):
                _badge = "watching"
            if _badge:
                try:
                    asyncio.create_task(ctx.room.local_participant.publish_data(
                        json.dumps({"kind": "badge", "s": _badge}).encode("utf-8"),
                        reliable=True))
                except Exception:
                    pass
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
                state._last_nova_at = time.time()   # FIX 4: silence-driver clock
                state._typed_reply_pending = 0      # her reply landed — clips unblocked
                logger.info(f"[SPEAK] Nova said → '{txt}'")
                _scan_nova_line(state, txt)   # light the organ she named / open the game she called
                # Save Nova's reply to history — multi-turn memory survives
                try:
                    memory.store.add_message(state.kid_id, "assistant", txt)
                except Exception as e:
                    logger.warning(f"[memory] add_message(assistant) failed: {e}")
            elif role == "user" and txt:
                if not txt.startswith("("):     # synthetic nudges aren't the kid talking
                    state._last_kid_at = time.time()   # FIX 4: silence-driver clock
                logger.info(f"[HEAR] confirmed user msg → '{txt}'")
    except Exception as e:
        logger.warning(f"[hook] conversation_item_added unavailable: {e}")

    # Runway face plugin — VOICE-ONLY FALLBACK (2026-07-02): if the avatar can't start
    # (e.g. Runway 400 "not enough credits", outage), DO NOT crash the job — Nova
    # continues as voice-only. Her audio publishes directly via RoomIO. The browser
    # reveals with a static face when only audio arrives. She must never fully die
    # because the face vendor is down.
    if voice_only:
        # ?voiceonly session (room metadata): Nova is VOICE-ONLY by request — no
        # avatar, no credits burned. Audio publishes via RoomIO; browser shows static face.
        logger.info("[nova-v207] step 2: VOICE-ONLY session (requested) → avatar skipped")
    elif _lemon_key() and os.getenv("NOVA_AVATAR", "lemonslice").lower() != "runway":
        # LEMONSLICE AVATAR (2026-07-09, user call: "$10 on LemonSlice, Runway on
        # standby"). Same LiveKit avatar contract as Runway — her voice re-routes
        # through the avatar participant, which lip-syncs and publishes video.
        # response_done_timeout=0.8 is REQUIRED for S2S voices (Gemini Live) per
        # LemonSlice docs — without it end-of-response detection stalls.
        # NOVA_AVATAR=runway flips back to the Runway path without a code change.
        _prev_audio_out = session.output.audio
        try:
            from livekit.plugins import lemonslice
            lemon_avatar = lemonslice.AvatarSession(
                agent_id=os.getenv("NOVA_LEMON_AGENT_ID", "agent_0a645f26d6d77246"),
                api_key=_lemon_key(),   # STAGE 1: key resolved from any env spelling
                response_done_timeout=0.8,
            )
            # BOUNDED like Runway: a hanging start sits BEFORE session.start and
            # would boot her voice late too. 20s (start includes an HTTP call).
            await asyncio.wait_for(lemon_avatar.start(session, room=ctx.room), timeout=20.0)
            logger.info("[nova-v207] step 2: LemonSlice avatar started "
                        f"(agent {os.getenv('NOVA_LEMON_AGENT_ID', 'agent_0a645f26d6d77246')[:18]})")
        except Exception as e:
            # CRITICAL: lemonslice.start() rebinds session audio to the avatar
            # BEFORE its HTTP call — on failure her voice points at a ghost
            # participant and every reply is silent. Restore the pre-start output
            # so RoomIO publishes her voice directly (true voice-only fallback).
            try:
                session.output.audio = _prev_audio_out
                logger.info("[nova-v207] audio output RESTORED after LemonSlice failure")
            except Exception:
                logger.exception("[nova-v207] audio restore failed — voice may be silent")
            if isinstance(e, asyncio.TimeoutError):
                logger.error("[nova-v207] LemonSlice start TIMED OUT (20s) → VOICE-ONLY fallback")
            else:
                logger.exception(f"[nova-v207] LemonSlice start FAILED → VOICE-ONLY fallback: {e}")
    else:
        try:
            runway_avatar = runway.AvatarSession(avatar_id=avatar_id)
            # BOUNDED (2026-07-03): a hanging Runway start (e.g. mid-credit-outage) used to
            # block HERE for ~60s — which sits BEFORE session.start, so her VOICE booted 55s
            # late too (the black-screen + late-greeting session). 15s and we move on.
            await asyncio.wait_for(runway_avatar.start(session, room=ctx.room), timeout=15.0)
            logger.info(f"[nova-v207] step 2: runway avatar started, id={avatar_id[:8]}")
        except asyncio.TimeoutError:
            logger.error("[nova-v207] runway start TIMED OUT (15s) → VOICE-ONLY fallback; voice pipeline proceeds NOW")
        except Exception as e:
            logger.exception(f"[nova-v207] runway start FAILED → VOICE-ONLY fallback (no avatar): {e}")

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
                # NOTE (2026-07-07): close_on_disconnect=False was tried here and
                # REVERTED the same hour — with it set, EVERY generate_reply timed
                # out (15s, no EVI response) while a direct Hume ws test was fine.
                # The reconnect problem is handled by the grace timer on
                # participant_disconnected instead.
            ),
        )
        logger.info("[nova-v207] step 5: session.start COMPLETE (kid-audio + text subscribed)")
    except Exception as e:
        logger.exception(f"[nova-v207] CRASH at session.start: {e}")
        raise

    logger.info("[nova-v207] step 6: pipeline ready, heavy logging active")
    # FIX 1 (2026-07-02): tell the browser Nova is TRULY ready (session started, voice
    # pipeline up) — the face reveal waits for THIS, not just for tracks arriving.
    # PHASE 1: RESEND every 2s until the browser acks with client-ready — a single
    # packet can land before the browser's handler is up (seen in loop-round-1).
    async def _announce_ready():
        # FAKE-READY KILL (2026-07-03): nova-ready used to mean "session.start returned" —
        # but the EVI websocket connects ASYNC after that, so the browser revealed a Nova
        # who couldn't speak yet (the mute sessions). Wait for the ws to be TRULY open.
        model = getattr(state, "_evi_model", None)
        if model is not None and hasattr(model, "connected_evt"):
            try:
                await asyncio.wait_for(model.connected_evt.wait(), timeout=25.0)
                logger.info("[nova-evi] ws CONNECTED — announcing TRULY-live nova-ready")
            except asyncio.TimeoutError:
                logger.error("[nova-evi] ws NOT connected after 25s — announcing anyway; expect mute (watch fire_greeting)")
        # belt AND suspenders: participant ATTRIBUTES (robust, state-synced by LiveKit —
        # survives data-channel death, late joiners read it instantly) + data packets.
        try:
            _res = ctx.room.local_participant.set_attributes({"nova_ready": "1"})
            if asyncio.iscoroutine(_res):
                await _res
            logger.info("[nova-v207] nova_ready attribute set")
        except Exception as e:
            logger.warning(f"[nova-ready] set_attributes failed: {e}")
        for _ in range(15):                       # up to 30s of announcing
            if state.client_ready.is_set() or not state.active:
                return
            try:
                await ctx.room.local_participant.publish_data(
                    json.dumps({"kind": "nova-ready"}).encode("utf-8"), reliable=True)
                logger.info("[nova-v207] sent nova-ready → browser may reveal her face")
            except Exception as e:
                logger.warning(f"[nova-ready] publish failed: {e}")
            await asyncio.sleep(2.0)
    asyncio.create_task(_announce_ready())

    # GREETING — first words from OUR brain
    state.greeting_done = True
    logger.info("[nova-v207] step 7: about to generate greeting...")

    # Rebuild sprint: HARDCODED first greeting — no LLM wait at session start
    # (saves ~2-3s, consistent first impression). The LLM drives every reply
    # AFTER this opening line.
    # PHASE 1: if the pre-reveal vision snapshot returned a detail, she OPENS with
    # proof she sees ("ohh — I love that purple shirt!"). SKIP/None → clean greet.
    _vis = (state.ctx.observed_visual or "").strip()
    _vis_ok = _vis and len(_vis) < 60 and "skip" not in _vis.lower() and "could not" not in _vis.lower()
    if state.ctx.name and state.ctx.sessions_before > 0:
        first_line = f"ohh — {state.ctx.name}! you came BACK! ...ready to play again?"
    elif _vis_ok:
        _v = _vis if _vis.lower().startswith(("the ", "that ", "your ")) else f"that {_vis}"
        first_line = f"ohh — I love {_v}! hi! I'm Nova — your magic movement friend! ...what's your name?"
    else:
        first_line = "hi! I'm Nova — your magic movement friend! ...what's your name?"

    # INTRO-FINAL (2026-07-03): NO greeting trigger here. ONE AUTHORITY — the browser's
    # reveal-now packet (handled in register_data_handler) fires the greeting: EVI →
    # model.fire_greeting(); non-EVI → _nova_say(state._first_line). The old client-ready
    # wait + 12s fallback were the second clock that let her speak 8s before her face.
    state._first_line = first_line
    logger.info("[nova-v207] step 8: greeting armed — waits for reveal-now (single authority)")

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
        # COMMERCIAL-INTRO Part A.5/E: dead air >4s in her court = ERROR, named cause
        asyncio.create_task(_dead_air_watch(state))

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
def _keepwarm_loop():
    """KEEP-WARM (2026-07-06): this background worker NEVER sleeps — it keeps the free
    web service hot. GitHub's '*/5' cron actually fires ~hourly (free-tier throttling,
    measured) while Render sleeps at 15 idle minutes → 52s cold-start at tap. A ping
    from here every 4 minutes closes that hole for good."""
    import threading as _t  # noqa: F401 (documentation: started as a daemon thread)
    import urllib.request
    url = os.getenv("NOVA_WEB_HEALTH_URL", "https://novapython.onrender.com/health")
    while True:
        try:
            urllib.request.urlopen(url, timeout=20).read()
        except Exception:
            pass
        time.sleep(240)


if __name__ == "__main__":
    import threading
    threading.Thread(target=_keepwarm_loop, daemon=True, name="nova-keepwarm").start()
    logger.info("[keepwarm] worker-side web pinger started (240s cadence)")
    logger.info("[V2V] flag NOVA_V2V=" + ("ON" if _v2v_on() else "OFF (today's build)"))
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            # Named agent → server explicitly dispatches "nova" into each room.
            # Without this, newer LiveKit won't auto-route the worker to rooms.
            agent_name="nova",
        )
    )
