"""
GEMINI LIVE FALLBACK VOICE (2026-07-08, user call: "use it, it's ok for now").

Hume ran out of credits mid-test (3rd time) — this is the insurance voice:
USE_GEMINI=1 swaps the mouth+ears to Gemini Live realtime (the same plugin the
avatar-hostess project runs in production) WITHOUT touching the Hume path or
the whisper/clip/turn architecture. agent.py drives every voice through one
control surface (push_context / fire_greeting / send_user_text / _ears_open /
_mouth_hold); this adapter maps that surface onto an AgentSession running
google.realtime.RealtimeModel.

Known "ok for now" tradeoffs (user-accepted):
- Clips stay in Kora's voice; Gemini live voice differs (voice mismatch).
- Gemini has no turn-less context channel — whispers are buffered and ride the
  NEXT directed generation instead of landing instantly.
"""
import asyncio
import logging
import os

logger = logging.getLogger("nova-gemini")


class GeminiVoiceAdapter:
    """The _evi_model-shaped control surface, bound to an AgentSession whose
    llm is google.realtime.RealtimeModel."""

    def __init__(self, system_prompt: str | None = None):
        self.connected_evt = asyncio.Event()
        self._session = None                # AgentSession — bound after start
        self._system_prompt = system_prompt
        self._greet_fired = False
        self._greet_text_override = None
        self._mouth_hold = False
        self._clip_playing = False
        self.__ears_open = False
        self._ctx_notes: list[str] = []     # whispers ride the next generation
        self.last_error = None
        self._inflight = 0                  # generate_reply calls still awaiting
        self._pending_turn = None           # kid turn that arrived mid-generation
        self._pending_multi = False         # 2+ turns merged while queued
        self._announce = None               # set by agent.py — publishes diag into the room

    def bind(self, session) -> None:
        self._session = session
        self._apply_ears()
        self.connected_evt.set()
        logger.info("[gemini] adapter bound to AgentSession (voice live)")

    # ── EARS: agent.py toggles _ears_open like a field; the setter maps it to
    # the session's real audio-input gate (clips close it, reveal opens it). ──
    @property
    def _ears_open(self) -> bool:
        return self.__ears_open

    @_ears_open.setter
    def _ears_open(self, v: bool) -> None:
        self.__ears_open = bool(v)
        self._apply_ears()

    def _apply_ears(self) -> None:
        s = self._session
        if s is None:
            return
        try:
            # SPEECH-SHIELD (2026-07-17 flicker): the door composes the owner's
            # verdict with a short shield around each line's BIRTH — noise that
            # arrives in the gen→playback gap was beheading every line (friend
            # mode keeps the door open, server VAD truncated at the source).
            shielded = getattr(self, "_shield_until", 0.0) > time.time()
            eff = self.__ears_open and not shielded
            s.input.set_audio_enabled(eff)
            logger.info(f"[EARS] gemini door {'OPEN' if eff else 'CLOSED'}"
                        + (" (shield)" if self.__ears_open and shielded else ""))
        except Exception:
            logger.exception("[gemini] set_audio_enabled failed")

    def _shield(self, seconds: float, reason: str) -> None:
        """Close the door for `seconds` (composes with the owner). DEFAULT OFF
        (2026-07-17 quiet-room probe: closing the door stops audio FRAMES to the
        realtime API entirely → the server's turn state stalls and the response
        never plays — she went fully silent even in quiet rooms). NOVA_SPEECH_SHIELD=1
        re-arms it for experiments; the durable noise fix must send SILENCE frames,
        not stop frames (bridge-style), or clean the mic in the browser."""
        if os.getenv("NOVA_SPEECH_SHIELD", "0") != "1":
            return
        import time as _t
        self._shield_until = max(getattr(self, "_shield_until", 0.0), _t.time() + seconds)
        logger.info(f"[SHIELD] ear shielded {seconds:.1f}s — {reason}")
        self._apply_ears()

        async def _lift():
            await asyncio.sleep(seconds + 0.1)
            self._apply_ears()
        try:
            asyncio.create_task(_lift())
        except Exception:
            pass

    # ── WHISPERS: no session_settings.context on Gemini — buffer, deliver on
    # the next directed line so her awareness still arrives (delayed, not lost).
    def push_context(self, text: str) -> bool:
        self._ctx_notes.append(str(text))
        self._ctx_notes = self._ctx_notes[-6:]
        return True

    def _drain_notes(self) -> str:
        notes, self._ctx_notes = self._ctx_notes, []
        return " ".join(notes)

    def _diag(self, ev: str, **fields) -> None:
        """VOICE-SILENCE DEBUG (2026-07-09): announce generation lifecycle into
        the room so an external probe sees where a reply stalls."""
        logger.info(f"[gemini-diag] {ev} {fields}")
        cb = self._announce
        if cb is not None:
            try:
                cb({"kind": "gemini-diag", "ev": ev, **fields})
            except Exception:
                pass

    # ── MOUTH ──
    def fire_greeting(self, retry: bool = False) -> bool:
        if self._greet_fired and not retry:
            return False
        self._greet_fired = True
        self._mouth_hold = False
        text = ("(if you have NOT greeted yet: greet now. if you ALREADY greeted or are speaking "
                "right now: just warmly ask their name again, one short line, nothing else)"
                if retry else
                (self._greet_text_override
                 or "(the dancer just appeared on screen — greet them now)"))
        # the greet is PROTECTED: a 1ms noise blip must never behead her first line
        # (live session aa784c31: greet cut instantly → MUTE-ALARM → 75s first word)
        return self._reply(user_input=text, protect=not retry)

    def send_user_text(self, text: str, preempt: bool = False) -> bool:
        """Typed chat = a real kid turn. Whisper notes ride along as instructions.
        SPEECH-GUARD LAW (2026-07-16): preempt=True marks a beat-CHANGE (e.g. a
        celebration) — the only producer input allowed to cut her active line."""
        self._mouth_hold = False
        if self._inflight > 0:
            # TURN QUEUE (2026-07-10 calltuns): never DROP a turn because she is
            # mid-generation — hold the latest and answer the moment the current
            # line lands. Rule: no kid utterance goes unanswered.
            if getattr(self, "_pending_turn", None):
                self._pending_turn = f"{self._pending_turn} {text}"
                self._pending_multi = True   # PERSONA-FIX 6: 2+ merged turns
            else:
                self._pending_turn = text
            logger.info(f"[gemini] turn QUEUED (gen inflight): '{text[:50]}'")
            return True
        return self._reply(user_input=text, preempt=preempt)

    def clear_pending_stage(self) -> None:
        """PERSONA-FIX 5 (2026-07-10): a celebration must WIN the race against an
        already-queued staged line — drop the pending turn if it is purely a
        (stage direction); kid words are never dropped."""
        p = getattr(self, "_pending_turn", None)
        if p and p.lstrip().startswith("("):
            logger.info(f"[gemini] pending STAGED turn cleared for a preempting line: '{p[:50]}'")
            self._pending_turn = None
            self._pending_multi = False

    def _reply(self, user_input: str, preempt: bool = False, protect: bool = False) -> bool:
        s = self._session
        if s is None:
            logger.warning("[gemini] reply requested before bind — dropped")
            return False
        notes = self._drain_notes()
        kwargs = {"user_input": user_input}
        # ANTI-FLICKER (2026-07-17, live session aa784c31: her talking windows were
        # 1ms-600ms — every noise blip beheaded her, "I love"/"Loud" transcripts):
        # protect=True makes THIS line uninterruptible (used for the greet only —
        # one short line by speech law; kid audio still buffers, see agent.py
        # discard_audio_if_uninterruptible=False).
        if protect:
            try:
                import inspect as _insp
                if "allow_interruptions" in _insp.signature(s.generate_reply).parameters:
                    kwargs["allow_interruptions"] = False
            except Exception:
                pass
        if notes:
            kwargs["instructions"] = f"(your own awareness right now, never read aloud: {notes})"
        # HEBREW MODE (2026-07-15 probe: first 3 lines leaked English): EVERY turn
        # carries the language law — greet, briefs, silence nudges, all paths.
        _st_lang = getattr(getattr(getattr(self, "_state", None), "ctx", None), "lang", "en")
        if _st_lang == "he":
            _he_law = "(דברי עברית בלבד — אף מילה באנגלית, גם אם ההנחיה כתובה באנגלית)"
            kwargs["instructions"] = (kwargs.get("instructions", "") + " " + _he_law).strip()
        else:
            # LANGUAGE LOCK EN (2026-07-16, live session: the ear misheard the kid as
            # German and she ANSWERED in German): English sessions answer in English,
            # always — never mirror a misheard language.
            _en_law = "(speak English ONLY — even if the child seems to speak another language, answer warmly in English)"
            kwargs["instructions"] = (kwargs.get("instructions", "") + " " + _en_law).strip()

        async def _go():
            import time as _t
            # STAGE 1 MOUTH-GATE (FIX-EVERYTHING): never start a generation while
            # her audio is still PLAYING (playback clock — includes clips). Wait
            # up to 6s for air; still blocked → the request dies, logged.
            _st = getattr(self, "_state", None)
            if _st is not None:
                # DEFAULT MUST MATCH agent.py _friend_on() ("1" = ON): a "0"
                # default here left the mouth on the strict gate while the ear
                # ran friend law — two halves of the voice on different rules.
                _friend = os.getenv("NOVA_FRIEND", "1") == "1"
                # SPEECH-GUARD LAW (2026-07-16, "one bug" order): nothing may cancel
                # her ACTIVE utterance except a beat-CHANGE (preempt=True) or the kid
                # barging in by VOICE (native VAD path, not through here). Staged
                # nudges and typed turns QUEUE until her current line's playback ends.
                # The old friend-mode "interruptions are native" gate let a nudge start
                # a generation mid-word — the 06-18 log beheadings ("Look, right there
                # on your—" cut by the re-pulse beat, "Now—" cut by picker-open).
                def _busy():
                    if _friend and preempt:
                        # beat-change keeps the native-interrupt right (clips still block)
                        return getattr(_st, "_clip_playing", False) or self._clip_playing
                    _pending = (not getattr(_st, "_is_speaking", False)
                                and _t.time() < getattr(self, "_await_playback_until", 0.0))
                    return (getattr(_st, "_is_speaking", False)
                            or _pending
                            or getattr(_st, "_clip_playing", False)
                            or self._clip_playing)
                _cap = 6.0 if preempt else 12.0
                _w0 = _t.time()
                while _t.time() - _w0 < _cap and _busy():
                    await asyncio.sleep(0.15)
                if _busy():
                    if user_input.lstrip().startswith("("):
                        # a staged line that found no air in 12s dies — the producer re-decides
                        logger.info(f"[MOUTH-GATE] blocked: staged turn '{user_input[:40]}' — audio still playing after {_cap:.0f}s, dropped")
                        self.last_error = "mouth-gate blocked (audio playing)"
                        self._diag("gen-blocked", text=user_input[:50])
                        return
                    # kid words are never dropped — proceed even if late (their turn)
                    logger.info(f"[MOUTH-GATE] kid turn proceeds after {_cap:.0f}s wait: '{user_input[:40]}'")
            t0 = _t.time()
            self._inflight += 1
            # SPEECH-SHIELD: every line gets a short birth shield (noise in the
            # gen→playback gap beheaded lines at 1ms); a PROTECTED line (the
            # greet) holds the shield through its whole playback window.
            self._shield(7.0 if protect else 2.2,
                         "protected line" if protect else "line birth")
            self._diag("gen-start", text=user_input[:50], inflight=self._inflight)
            try:
                await s.generate_reply(**kwargs)
                self.last_error = None
                # SPEECH-GUARD race bridge (2026-07-16 probe: "Six and strong! Ready
                # to move those" beheaded): between gen-done and the audio actually
                # STARTING, _is_speaking is still False — the mouth-gate saw quiet and
                # let the next staged turn cut the buffered reply. Hold the gate busy
                # until playback begins (2.5s cap covers zero-audio replies).
                self._await_playback_until = _t.time() + 2.5
                self._diag("gen-done", text=user_input[:50], elapsed=round(_t.time() - t0, 2))
            except Exception as e:
                # VOICE-SILENCE DEBUG (2026-07-09): keep the failure text so the
                # worker can announce it into the room (Render logs are the only
                # other place this lands, and they're often out of reach)
                self.last_error = f"{type(e).__name__}: {e}"
                self._diag("gen-error", text=user_input[:50], err=self.last_error[:200],
                           elapsed=round(_t.time() - t0, 2))
                logger.exception("[gemini] generate_reply failed")
            finally:
                self._inflight -= 1
                # TURN QUEUE drain: a kid turn arrived while this line was being
                # made — answer it NOW (latest wins; stale intermediates merged).
                _p, self._pending_turn = getattr(self, "_pending_turn", None), None
                _multi, self._pending_multi = getattr(self, "_pending_multi", False), False
                if _p and self._inflight == 0:
                    if _multi:
                        # PERSONA-FIX 6: rapid-fire questions collapsed into one
                        # answer ("are you real??" was dropped) — tell her to
                        # touch EACH thing, still in one short warm turn.
                        self._ctx_notes.append("they said several things at once — "
                                               "answer EACH one briefly, one warm line total")
                    logger.info(f"[gemini] draining queued turn: '{str(_p)[:50]}'")
                    self._reply(user_input=_p)
        asyncio.create_task(_go())
        logger.info(f"[EVI->] gemini user turn: '{user_input[:60]}'"
                    + (" (+awareness notes)" if notes else ""))
        return True
