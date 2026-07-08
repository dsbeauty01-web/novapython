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
            s.input.set_audio_enabled(self.__ears_open)
            logger.info(f"[EARS] gemini door {'OPEN' if self.__ears_open else 'CLOSED'}")
        except Exception:
            logger.exception("[gemini] set_audio_enabled failed")

    # ── WHISPERS: no session_settings.context on Gemini — buffer, deliver on
    # the next directed line so her awareness still arrives (delayed, not lost).
    def push_context(self, text: str) -> bool:
        self._ctx_notes.append(str(text))
        self._ctx_notes = self._ctx_notes[-6:]
        return True

    def _drain_notes(self) -> str:
        notes, self._ctx_notes = self._ctx_notes, []
        return " ".join(notes)

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
        return self._reply(user_input=text)

    def send_user_text(self, text: str) -> bool:
        """Typed chat = a real kid turn. Whisper notes ride along as instructions."""
        self._mouth_hold = False
        return self._reply(user_input=text)

    def _reply(self, user_input: str) -> bool:
        s = self._session
        if s is None:
            logger.warning("[gemini] reply requested before bind — dropped")
            return False
        notes = self._drain_notes()
        kwargs = {"user_input": user_input}
        if notes:
            kwargs["instructions"] = f"(your own awareness right now, never read aloud: {notes})"

        async def _go():
            try:
                await s.generate_reply(**kwargs)
            except Exception:
                logger.exception("[gemini] generate_reply failed")
        asyncio.create_task(_go())
        logger.info(f"[EVI->] gemini user turn: '{user_input[:60]}'"
                    + (" (+awareness notes)" if notes else ""))
        return True
