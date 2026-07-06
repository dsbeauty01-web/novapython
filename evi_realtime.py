from __future__ import annotations

"""
HumeEVIRealtimeModel — a custom livekit-agents RealtimeModel backed by Hume AI's
EVI (Empathic Voice Interface, EVI 3) speech-to-speech websocket.

Target: livekit-agents >=1.5.0,<2.0.0  (developed against the 1.5.17 abstract API:
github.com/livekit/agents tag `livekit-agents@1.5.17`,
livekit-agents/livekit/agents/llm/realtime.py).

Drop-in usage:
    AgentSession(llm=HumeEVIRealtimeModel())          # no stt / tts / vad needed
A Runway AvatarSession attached to the AgentSession will lip-sync the EVI audio,
because the avatar consumes the session's audio *output*, which for a realtime
model is the model's own audio.

Env:
    HUME_API_KEY      (required)
    HUME_SECRET_KEY   (optional; if set, we mint an OAuth access_token instead of
                       passing the api_key directly on the ws URL)
    HUME_CONFIG_ID    (optional; the EVI configuration UUID)

Docs used (Hume):
    https://dev.hume.ai/reference/speech-to-speech-evi/chat
    https://dev.hume.ai/docs/speech-to-speech-evi/overview
    https://dev.hume.ai/docs/speech-to-speech-evi/guides/audio
    https://dev.hume.ai/docs/introduction/api-key
    https://dev.hume.ai/docs/speech-to-speech-evi/configuration/session-settings

LiveKit source used (tag livekit-agents@1.5.17):
    .../livekit/agents/llm/realtime.py      (abstract RealtimeModel/RealtimeSession)
    .../livekit/agents/llm/__init__.py      (export paths)
    .../livekit-plugins-openai/.../realtime/realtime_model.py  (template)
"""

import asyncio
import base64
import io
import json
import os
import wave
import weakref
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlencode

import aiohttp

from livekit import rtc

# All of these names are re-exported from livekit.agents.llm (verified in
# llm/__init__.py at tag livekit-agents@1.5.17).
from livekit.agents import llm, utils
from livekit.agents.llm import (
    ChatContext,
    GenerationCreatedEvent,
    InputSpeechStartedEvent,
    InputSpeechStoppedEvent,
    InputTranscriptionCompleted,
    MessageGeneration,
    RealtimeCapabilities,
    RealtimeModel,
    RealtimeError,
    RealtimeSession,
    RealtimeSessionReconnectedEvent,
    Tool,
    ToolChoice,
    ToolContext,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

import logging

logger = logging.getLogger("hume-evi-realtime")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

HUME_WS_URL = "wss://api.hume.ai/v0/evi/chat"
HUME_TOKEN_URL = "https://api.hume.ai/oauth2-cc/token"

# Format of the headerless Linear16 PCM we declare to EVI in session_settings and
# send via audio_input. EVI accepts "linear16" mono at the rate we declare.
INPUT_SAMPLE_RATE = 16000
INPUT_NUM_CHANNELS = 1

# Format LiveKit / the Runway avatar expect on the session output. The avatar
# lip-syncs whatever audio frames we push downstream; 48k mono is the LiveKit norm.
OUTPUT_SAMPLE_RATE = 48000
OUTPUT_NUM_CHANNELS = 1

# How much input audio we batch per audio_input message (~100 ms).
_INPUT_CHUNK_SAMPLES = INPUT_SAMPLE_RATE // 10


# --------------------------------------------------------------------------- #
# Internal generation bookkeeping (mirrors the OpenAI plugin's _ResponseGeneration
# / _MessageGeneration helpers).
# --------------------------------------------------------------------------- #


@dataclass
class _MessageGeneration:
    message_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    modalities: asyncio.Future  # asyncio.Future[list[Literal["text", "audio"]]]


@dataclass
class _ResponseGeneration:
    message_ch: utils.aio.Chan[MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]
    messages: dict[str, _MessageGeneration]


# --------------------------------------------------------------------------- #
# RealtimeModel
# --------------------------------------------------------------------------- #


class HumeEVIRealtimeModel(RealtimeModel):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        secret_key: str | None = None,
        config_id: str | None = None,
        config_version: int | None = None,
        system_prompt: str | None = None,
        variables: dict[str, str] | None = None,
        voice_id: str | None = None,
        input_sample_rate: int = INPUT_SAMPLE_RATE,
        output_sample_rate: int = OUTPUT_SAMPLE_RATE,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        # EVI does its own server-side VAD, turn-taking, interruption detection,
        # and transcription. So:
        #   turn_detection=True       -> we emit input_speech_started/stopped
        #   user_transcription=True   -> we emit input_audio_transcription_completed
        #   audio_output=True         -> we produce audio directly (avatar consumes it)
        #   message_truncation=False  -> EVI has no per-message truncate API
        super().__init__(
            capabilities=RealtimeCapabilities(
                message_truncation=False,
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=False,
                audio_output=True,
                manual_function_calls=False,
                mutable_chat_context=False,
                mutable_instructions=False,
                mutable_tools=False,
                per_response_tool_choice=False,
                supports_say=False,
            )
        )

        self._api_key = api_key or os.environ.get("HUME_API_KEY")
        if not self._api_key:
            raise ValueError("HUME_API_KEY is required (pass api_key= or set env)")
        self._secret_key = secret_key or os.environ.get("HUME_SECRET_KEY")
        self._config_id = config_id or os.environ.get("HUME_CONFIG_ID")
        self._config_version = config_version

        self._system_prompt = system_prompt
        # W0112 ROOT-CAUSE FIX: when a per-session prompt is given it is LOCKED —
        # update_instructions() must not overwrite it with the framework's phase
        # prompts (their wording trips Hume's content filter and poisons every
        # (re)connect for the whole session).
        self._prompt_locked = system_prompt is not None
        self._variables = variables
        self._voice_id = voice_id
        self._greet_fired = False   # INTRO-FINAL: set by fire_greeting(); guards the watchdog + dedupe
        self.connected_evt: asyncio.Event = asyncio.Event()   # set the moment the EVI ws is truly open

        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate

        self._http_session = http_session
        self._http_session_owned = False

        self._sessions = weakref.WeakSet["HumeEVIRealtimeSession"]()

    @property
    def model(self) -> str:
        return "evi-3"

    @property
    def provider(self) -> str:
        return "hume"

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            try:
                self._http_session = utils.http_context.http_session()
            except RuntimeError:
                self._http_session = aiohttp.ClientSession()
                self._http_session_owned = True
        return self._http_session

    def fire_greeting(self, retry: bool = False) -> bool:
        """INTRO-FINAL: THE one and only greeting trigger — called by the worker's
        reveal-now handler. Nudges EVI to greet (her brain speaks per the calm-opening
        prompt). Dedupes: returns False if already fired.

        retry=True sends a SELF-GUARDING nudge: if the first nudge actually landed
        (cold first-gen raced the mute guard — live 2026-07-04: the duplicate became a
        phantom user turn and she said "ohh — cool name!" to nobody), the worst a
        duplicate can produce is a warm re-ask of the name, never a fake reaction."""
        if self._greet_fired and not retry:
            return False
        self._greet_fired = True
        text = ("(if you have NOT greeted yet: greet now. if you ALREADY greeted or are speaking "
                "right now: just warmly ask their name again, one short line, nothing else)"
                if retry else
                (getattr(self, "_greet_text_override", None)
                 or "(the dancer just appeared on screen — greet them now)"))
        fired = False
        for sess in list(self._sessions):
            try:
                sess._send({"type": "user_input", "text": text})
                fired = True
            except Exception:
                logger.exception("fire_greeting: send failed")
        logger.info(f"[REVEAL] fire_greeting → sent={fired}{' (retry)' if retry else ''}")
        return fired

    def push_context(self, text: str) -> bool:
        """CONTEXT SYNC (2026-07-06): clip lines play in the BROWSER and never enter
        EVI's chat — her brain didn't know what 'she' just said, so replies felt
        disconnected ('she doesn't listen'). Hume session_settings.context injects
        text into her awareness WITHOUT speaking it. Temporary = rides the next turn."""
        ok = False
        for sess in list(self._sessions):
            try:
                sess._send({"type": "session_settings",
                            "context": {"text": text, "type": "temporary"}})
                ok = True
            except Exception:
                logger.exception("push_context: send failed")
        return ok

    def session(self) -> "HumeEVIRealtimeSession":
        sess = HumeEVIRealtimeSession(self)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None:
        for sess in list(self._sessions):
            await sess.aclose()
        if self._http_session_owned and self._http_session:
            await self._http_session.close()


# --------------------------------------------------------------------------- #
# RealtimeSession
# --------------------------------------------------------------------------- #


class HumeEVIRealtimeSession(
    RealtimeSession[Literal["hume_server_message_received", "hume_client_message_queued"]]
):
    """A LiveKit RealtimeSession backed by one Hume EVI websocket connection.

    Extra (non-standard) events you can listen to for debugging:
      - "hume_server_message_received": the raw decoded dict from EVI
      - "hume_client_message_queued":   the raw dict we send to EVI
    """

    def __init__(self, realtime_model: HumeEVIRealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model: HumeEVIRealtimeModel = realtime_model

        self._chat_ctx = ChatContext.empty()
        self._tools = ToolContext.empty()

        # outbound queue of dict messages -> websocket
        self._msg_ch = utils.aio.Chan[dict[str, Any]]()

        # input resampler (LiveKit frame rate -> EVI input rate, mono)
        self._input_resampler: rtc.AudioResampler | None = None
        # 100 ms byte batching at the EVI input rate
        self._bstream = utils.audio.AudioByteStream(
            self._realtime_model._input_sample_rate,
            INPUT_NUM_CHANNELS,
            samples_per_channel=max(1, self._realtime_model._input_sample_rate // 10),
        )
        # output resampler (EVI WAV rate -> LiveKit/Runway 48k mono); rebuilt per
        # source rate because each EVI chunk is a self-contained WAV.
        self._output_resampler: rtc.AudioResampler | None = None
        self._output_resampler_input_rate: int | None = None

        self._current_generation: _ResponseGeneration | None = None
        # message_id we attach EVI audio_output / assistant_message to within a turn
        self._active_message_id: str | None = None
        # set by generate_reply(); resolved when the next assistant turn starts
        self._pending_user_initiated_fut: asyncio.Future | None = None

        self._closed = False
        self._reconnect_event = asyncio.Event()

        self._main_atask = asyncio.create_task(
            self._main_task(), name="HumeEVIRealtimeSession._main_task"
        )

    # ----------------------- properties / context ------------------------- #

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat_ctx.copy()

    @property
    def tools(self) -> ToolContext:
        return self._tools.copy()

    async def update_instructions(self, instructions: str) -> None:
        # EVI's system prompt is fixed for the connection (set via session_settings
        # / config). We store it; it takes effect on (re)connect.
        rt = self._realtime_model
        if getattr(rt, "_prompt_locked", False):
            # Per-session EVI prompt (Hume-safe wording) is authoritative — the
            # framework phase prompts get blocked by Hume's filter (W0112).
            logger.debug("update_instructions ignored — per-session EVI prompt is locked")
            return
        rt._system_prompt = instructions

    async def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        # EVI manages conversation state server-side. We just track it locally.
        self._chat_ctx = chat_ctx.copy()

    async def update_tools(self, tools: list[Tool]) -> None:
        self._tools = ToolContext(tools)

    def update_options(self, *, tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN) -> None:
        # EVI tool_choice is config-driven; no-op for the realtime websocket.
        pass

    # --------------------------- input audio ------------------------------ #

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        """Receive an input audio frame from LiveKit, resample to EVI's input
        format, batch into ~100ms chunks, and forward as base64 audio_input."""
        if self._closed:
            return
        for f in self._resample_input(frame):
            for nf in self._bstream.write(f.data.tobytes()):
                self._send(
                    {
                        "type": "audio_input",
                        "data": base64.b64encode(nf.data.tobytes()).decode("utf-8"),
                    }
                )

    def _resample_input(self, frame: rtc.AudioFrame):
        target = self._realtime_model._input_sample_rate
        if self._input_resampler is not None:
            if frame.sample_rate != self._input_resampler._input_rate:
                self._input_resampler = None
        if self._input_resampler is None and (
            frame.sample_rate != target or frame.num_channels != INPUT_NUM_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=target,
                num_channels=INPUT_NUM_CHANNELS,
            )
        if self._input_resampler is not None:
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    def push_video(self, frame: rtc.VideoFrame) -> None:
        # EVI is audio-only; ignore video.
        pass

    def commit_audio(self) -> None:
        # EVI uses continuous server-side VAD; there is no explicit input-buffer
        # commit. No-op.
        pass

    def clear_audio(self) -> None:
        # No client-side input buffer to clear with EVI's streaming model.
        pass

    # --------------------------- generation ------------------------------- #

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        user_input: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[GenerationCreatedEvent]:
        """Ask EVI to speak. EVI normally drives turns itself from the audio
        stream; this path is for an explicit, programmatic prompt.
        instructions= → she speaks the text VERBATIM (assistant_input).
        user_input=   → her brain RESPONDS to the text (typed chat path —
                        was silently unsupported before 2026-07-03; typing got no reply)."""
        fut: asyncio.Future[GenerationCreatedEvent] = asyncio.Future()

        text = instructions if is_given(instructions) else ""
        if text:
            # `assistant_input` makes EVI speak the given text directly.
            self._send({"type": "assistant_input", "text": text})
        elif is_given(user_input) and user_input:
            # typed chat: EVI treats it exactly like a spoken kid line.
            self._send({"type": "user_input", "text": user_input})
        else:
            # nudge: ask EVI to continue. With no text we simply mark user_initiated.
            pass

        # The actual GenerationCreatedEvent is resolved when EVI starts emitting
        # the assistant turn (see _start_generation). Stash the future.
        self._pending_user_initiated_fut = fut

        def _on_timeout() -> None:
            if not fut.done():
                fut.set_exception(RealtimeError("generate_reply timed out."))

        handle = asyncio.get_event_loop().call_later(15.0, _on_timeout)
        fut.add_done_callback(lambda _f: handle.cancel())
        return fut

    def interrupt(self) -> None:
        """Stop the current EVI assistant output. EVI itself emits
        user_interruption when it detects barge-in; this also lets the agent
        framework interrupt explicitly."""
        self._send({"type": "pause_assistant_message"})
        self._finish_generation()
        # immediately allow EVI to speak again on the next turn
        self._send({"type": "resume_assistant_message"})

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        # capabilities.message_truncation is False; nothing to do.
        pass

    # ------------------------------ close --------------------------------- #

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._finish_generation()
        if not self._msg_ch.closed:
            self._msg_ch.close()
        if self._main_atask:
            await utils.aio.cancel_and_wait(self._main_atask)

    # ======================================================================= #
    # internals
    # ======================================================================= #

    def _send(self, msg: dict[str, Any]) -> None:
        if self._closed or self._msg_ch.closed:
            return
        self.emit("hume_client_message_queued", msg)
        try:
            self._msg_ch.send_nowait(msg)
        except utils.aio.ChanClosed:
            pass

    def _build_ws_url(self, access_token: str | None) -> str:
        params: dict[str, str] = {}
        if access_token:
            params["access_token"] = access_token
        else:
            params["api_key"] = self._realtime_model._api_key  # type: ignore[assignment]
        if self._realtime_model._config_id:
            params["config_id"] = self._realtime_model._config_id
        if self._realtime_model._config_version is not None:
            params["config_version"] = str(self._realtime_model._config_version)
        return f"{HUME_WS_URL}?{urlencode(params)}"

    async def _mint_access_token(self) -> str | None:
        """If a secret key is configured, exchange api_key+secret for an OAuth
        access token (recommended). Otherwise return None and use api_key directly."""
        rt = self._realtime_model
        if not rt._secret_key:
            return None
        # Mint an OAuth token; if anything fails, fall back to the api_key query param
        # (EVI accepts ?api_key=... directly), so auth never hard-blocks the connection.
        try:
            creds = base64.b64encode(f"{rt._api_key}:{rt._secret_key}".encode()).decode()
            session = rt._ensure_http_session()
            async with session.post(
                HUME_TOKEN_URL,
                headers={"Authorization": f"Basic {creds}"},
                data={"grant_type": "client_credentials"},
            ) as resp:
                resp.raise_for_status()
                # Hume's token endpoint returns JSON WITHOUT a json content-type header,
                # so aiohttp's strict resp.json() raises ContentTypeError → content_type=None.
                body = await resp.json(content_type=None)
                return body.get("access_token")
        except Exception as e:
            logger.warning(f"EVI token mint failed ({e}); using api_key directly")
            return None

    def _session_settings_msg(self) -> dict[str, Any]:
        rt = self._realtime_model
        msg: dict[str, Any] = {
            "type": "session_settings",
            "audio": {
                "encoding": "linear16",
                "sample_rate": rt._input_sample_rate,
                "channels": INPUT_NUM_CHANNELS,
            },
        }
        if rt._system_prompt:
            msg["system_prompt"] = rt._system_prompt
        if rt._variables:
            msg["variables"] = rt._variables
        if rt._voice_id:
            msg["voice_id"] = rt._voice_id
        return msg

    async def _main_task(self) -> None:
        """Connect (with reconnect) and run the send/recv loops."""
        rt = self._realtime_model
        while not self._closed:
            try:
                access_token = await self._mint_access_token()
                url = self._build_ws_url(access_token)
                session = rt._ensure_http_session()
                ws = await session.ws_connect(url)
            except Exception:
                logger.exception("EVI websocket connect failed; retrying in 2s")
                await asyncio.sleep(2.0)
                continue

            logger.info("connected to Hume EVI")
            self._realtime_model.connected_evt.set()   # gate item 'evi' = THIS, truly live
            # session_settings must be sent first (sets the input audio format).
            await ws.send_str(json.dumps(self._session_settings_msg()))
            # INTRO-FINAL (2026-07-03): NO greeting here. ONE AUTHORITY — the browser's
            # reveal-now packet is the only thing that fires the greeting (agent.py calls
            # model.fire_greeting()). Voice-before-face is now impossible, not "checked".
            # VOICE WATCHDOG (FIX 4): a mid-session reconnect (EVI ws died while the kid
            # is live) is logged LOUDLY and she resumes the beat out loud. Max 2 resumes.
            if getattr(self._realtime_model, "_greet_fired", False):
                n = getattr(self, "_reconnects", 0) + 1
                self._reconnects = n
                logger.error(f"[VOICE-WATCHDOG] EVI reconnected MID-SESSION (#{n}) — voice had died")
                if n <= 2:
                    async def _resume(w=ws):
                        await asyncio.sleep(0.8)
                        try:
                            await w.send_str(json.dumps({"type": "assistant_input", "text": "okay — where were we!"}))
                            logger.info("[VOICE-WATCHDOG] resume line sent")
                        except Exception as e:
                            logger.warning(f"[VOICE-WATCHDOG] resume failed: {e}")
                    asyncio.create_task(_resume())
                else:
                    logger.error("[VOICE-WATCHDOG] reconnect cap reached — captions must carry the intro")

            send_task = asyncio.create_task(self._send_task(ws), name="evi_send")
            recv_task = asyncio.create_task(self._recv_task(ws), name="evi_recv")
            try:
                done, _ = await asyncio.wait(
                    [send_task, recv_task], return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                for t in (send_task, recv_task):
                    if not t.done():
                        await utils.aio.cancel_and_wait(t)
                await ws.close()
                self._finish_generation()

            if self._closed:
                break

            # unexpected disconnect -> reconnect and notify the agent framework
            self._realtime_model.connected_evt.clear()   # not live until the next connect
            logger.warning("EVI websocket disconnected; reconnecting")
            self.emit("session_reconnected", RealtimeSessionReconnectedEvent())
            await asyncio.sleep(0.5)

    async def _send_task(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        async for msg in self._msg_ch:
            await ws.send_str(json.dumps(msg))

    async def _recv_task(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        async for ws_msg in ws:
            if ws_msg.type == aiohttp.WSMsgType.TEXT:
                self._handle_server_message(json.loads(ws_msg.data))
            elif ws_msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.ERROR,
            ):
                return

    # ----------------------- server message routing ----------------------- #

    def _handle_server_message(self, msg: dict[str, Any]) -> None:
        self.emit("hume_server_message_received", msg)
        mtype = msg.get("type")

        if mtype == "chat_metadata":
            logger.debug("EVI chat_metadata: %s", msg.get("chat_id"))

        elif mtype == "user_message":
            self._handle_user_message(msg)

        elif mtype == "assistant_message":
            self._handle_assistant_message(msg)

        elif mtype == "audio_output":
            self._handle_audio_output(msg)

        elif mtype == "user_interruption":
            self._handle_user_interruption(msg)

        elif mtype == "assistant_end":
            self._finish_generation()

        elif mtype == "error":
            logger.error("EVI error: %s", msg)
            # Hume's prompt filter is an LLM moderator and intermittently blocks
            # even clean prompts — a block means she silently runs on the console
            # fallback prompt (wrong script). Retry the send; the same text
            # usually passes on the next attempt.
            if msg.get("code") == "W0112" or "content_filter" in (msg.get("slug") or ""):
                n = getattr(self, "_prompt_retries", 0)
                if n < 2:
                    self._prompt_retries = n + 1
                    logger.error(f"[EVI-PROMPT] system_prompt BLOCKED by Hume filter — resend {n + 1}/2 "
                                 "(console fallback prompt is live until this lands)")

                    async def _resend() -> None:
                        await asyncio.sleep(1.5)
                        self._send(self._session_settings_msg())

                    asyncio.create_task(_resend())
                else:
                    logger.error("[EVI-PROMPT] prompt still blocked after 2 resends — "
                                 "session is stuck on the Hume console prompt")

        # tool_call / assistant_prosody etc. are ignored in this draft.

    def _handle_user_message(self, msg: dict[str, Any]) -> None:
        # EVI's server-side VAD has produced (an interim or final) user transcript.
        content = (msg.get("message") or {}).get("content", "")
        interim = bool(msg.get("interim", False))
        item_id = msg.get("id") or utils.shortuuid("user_")

        # Treat arrival of a user message as the user having spoken: surface
        # speech start/stop for turn-taking metrics. (EVI itself drives turns.)
        if not interim:
            self.emit("input_speech_started", InputSpeechStartedEvent())
            self.emit(
                "input_speech_stopped",
                InputSpeechStoppedEvent(user_transcription_enabled=True),
            )

        self.emit(
            "input_audio_transcription_completed",
            InputTranscriptionCompleted(
                item_id=item_id,
                transcript=content,
                is_final=not interim,
            ),
        )

    def _handle_assistant_message(self, msg: dict[str, Any]) -> None:
        # transcript of what EVI is about to / is speaking
        content = (msg.get("message") or {}).get("content", "")
        gen = self._ensure_generation()
        item_gen = self._ensure_message(gen, msg.get("id"))
        if content and not item_gen.text_ch.closed:
            item_gen.text_ch.send_nowait(content)

    def _handle_audio_output(self, msg: dict[str, Any]) -> None:
        gen = self._ensure_generation()
        item_gen = self._ensure_message(gen, self._active_message_id)

        # EVI audio_output.data is a base64-encoded, self-contained WAV file
        # (RIFF header + PCM). Parse the header so we honor EVI's actual output
        # rate, then resample to the LiveKit/Runway output rate.
        raw = base64.b64decode(msg["data"])
        try:
            with wave.open(io.BytesIO(raw), "rb") as wf:
                src_rate = wf.getframerate()
                src_ch = wf.getnchannels()
                pcm = wf.readframes(wf.getnframes())
        except (wave.Error, EOFError):
            logger.warning("EVI audio_output was not a valid WAV chunk; skipping")
            return

        src_frame = rtc.AudioFrame(
            data=pcm,
            sample_rate=src_rate,
            num_channels=src_ch,
            samples_per_channel=len(pcm) // (2 * src_ch),
        )
        for out in self._resample_output(src_frame, src_rate, src_ch):
            if not item_gen.audio_ch.closed:
                item_gen.audio_ch.send_nowait(out)

    def _resample_output(self, frame: rtc.AudioFrame, src_rate: int, src_ch: int):
        target = self._realtime_model._output_sample_rate
        if src_rate == target and src_ch == OUTPUT_NUM_CHANNELS:
            yield frame
            return
        if self._output_resampler is None or self._output_resampler_input_rate != src_rate:
            self._output_resampler = rtc.AudioResampler(
                input_rate=src_rate,
                output_rate=target,
                num_channels=OUTPUT_NUM_CHANNELS,
            )
            self._output_resampler_input_rate = src_rate
        # AudioResampler handles channel down-mix to num_channels=1.
        yield from self._output_resampler.push(frame)

    def _handle_user_interruption(self, msg: dict[str, Any]) -> None:
        # EVI detected the user barged in. Stop forwarding the current assistant
        # audio so the avatar stops talking immediately.
        logger.info("[BARGE-IN] EVI user_interruption → assistant audio cleared")
        self.emit("input_speech_started", InputSpeechStartedEvent())
        self._finish_generation()

    # --------------------- generation lifecycle helpers -------------------- #

    def _ensure_generation(self) -> _ResponseGeneration:
        if self._current_generation is not None:
            return self._current_generation
        return self._start_generation()

    def _start_generation(self) -> _ResponseGeneration:
        gen = _ResponseGeneration(
            message_ch=utils.aio.Chan(),
            function_ch=utils.aio.Chan(),
            messages={},
        )
        self._current_generation = gen

        user_initiated = False
        fut = self._pending_user_initiated_fut
        ev = GenerationCreatedEvent(
            message_stream=gen.message_ch,
            function_stream=gen.function_ch,
            user_initiated=user_initiated,
            response_id=utils.shortuuid("evi_resp_"),
        )
        if fut is not None and not fut.done():
            ev.user_initiated = True
            fut.set_result(ev)
            self._pending_user_initiated_fut = None

        self.emit("generation_created", ev)
        return gen

    def _ensure_message(
        self, gen: _ResponseGeneration, message_id: str | None
    ) -> _MessageGeneration:
        mid = message_id or self._active_message_id or utils.shortuuid("evi_msg_")
        self._active_message_id = mid
        if mid in gen.messages:
            return gen.messages[mid]

        modalities: asyncio.Future = asyncio.Future()
        modalities.set_result(["audio", "text"])
        item_gen = _MessageGeneration(
            message_id=mid,
            text_ch=utils.aio.Chan(),
            audio_ch=utils.aio.Chan(),
            modalities=modalities,
        )
        gen.messages[mid] = item_gen
        gen.message_ch.send_nowait(
            MessageGeneration(
                message_id=mid,
                text_stream=item_gen.text_ch,
                audio_stream=item_gen.audio_ch,
                modalities=item_gen.modalities,
            )
        )
        return item_gen

    def _finish_generation(self) -> None:
        gen = self._current_generation
        if gen is None:
            return
        self._current_generation = None
        self._active_message_id = None
        for item_gen in gen.messages.values():
            if not item_gen.text_ch.closed:
                item_gen.text_ch.close()
            if not item_gen.audio_ch.closed:
                item_gen.audio_ch.close()
        if not gen.function_ch.closed:
            gen.function_ch.close()
        if not gen.message_ch.closed:
            gen.message_ch.close()
