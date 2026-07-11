"""
POD VOICE OUTPUT — ORDER-LIPSYNC-TOPQUALITY (2026-07-11).

Her voice goes OUT through the self-hosted engine's video stream (one muxed
stream = frame-level lip-sync). This sink replaces the room audio publisher:
OpenAI realtime frames (24k PCM16) are streamed to the pod bridge's
/push_voice relay in ~0.5s chunks AS THEY ARRIVE (never buffer a whole
utterance — that's where integrations lose a second). The relay resamples to
16k (mandatory) and feeds /humanaudio; the FLV stream carries voice+lips back.

The app-side clocks this feeds:
- state._pod_audible_until — when her voice will have FINISHED coming out of
  the kid's speakers (push window + FLV delay). The ear owner and the gesture
  beats gate on it (echo guard: browser AEC cannot cancel MSE/FLV audio).
- on_playback_finished is reported on the NOMINAL audio duration (realtime
  simulation) so the AgentSession's turn logic stays sane.

Kill: NOVA_AVATAR != "pod" → this module is never bound.
"""
from __future__ import annotations

import os
import time
import asyncio
import logging

import aiohttp
from livekit import rtc
from livekit.agents.voice import io as voice_io

logger = logging.getLogger("nova-pod-voice")

PUSH_URL = os.getenv("NOVA_POD_VOICE_URL",
                     "https://u6m9xwo9s3l2iz-8765.proxy.runpod.net")
POD_SID = os.getenv("NOVA_POD_SID", "0")
CHUNK_S = float(os.getenv("NOVA_POD_CHUNK_S", "0.5"))
# FLV path delay: engine render + SRS + transcode + browser buffer (~0.9s)
FLV_DELAY_S = float(os.getenv("NOVA_POD_FLV_DELAY_S", "1.7"))
ECHO_MARGIN_S = float(os.getenv("NOVA_POD_ECHO_MARGIN_S", "1.0"))
SAMPLE_RATE = 24000  # OpenAI realtime output; relay resamples to 16k


class PodVoiceOutput(voice_io.AudioOutput):
    def __init__(self, state) -> None:
        super().__init__(
            label="PodVoice",
            capabilities=voice_io.AudioOutputCapabilities(pause=False),
            sample_rate=SAMPLE_RATE,
        )
        self._state = state
        self._buf = bytearray()
        self._http: aiohttp.ClientSession | None = None
        self._seg_dur = 0.0          # duration captured in the current segment
        self._seg_started = 0.0
        self._push_lock = asyncio.Lock()
        self._finish_task: asyncio.Task | None = None
        state._pod_audible_until = 0.0

    def _session(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession()
        return self._http

    async def _push(self, pcm: bytes) -> None:
        if not pcm:
            return
        dur = len(pcm) / 2 / SAMPLE_RATE
        try:
            async with self._push_lock:   # chunks must land in order
                async with self._session().post(
                    f"{PUSH_URL}/push_voice?rate={SAMPLE_RATE}&sid={POD_SID}&persona=dance",
                    data=pcm,
                    # explicit content-type — the RunPod proxy 403s the default
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as r:
                    body = await r.json(content_type=None)
                    remain = float(body.get("remain", dur))
                    # her voice is audible until push-window end + stream delay
                    self._state._pod_audible_until = (
                        time.time() + remain + FLV_DELAY_S + ECHO_MARGIN_S)
                    logger.info(f"[POD-VOICE] pushed {dur:.2f}s (remain {remain:.2f}s, "
                                f"audible +{remain + FLV_DELAY_S:.1f}s)")
        except Exception as e:
            logger.error(f"[POD-VOICE] push failed: {e}")

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        if not self._seg_dur:
            self._seg_started = time.time()
        self._buf.extend(bytes(frame.data))
        self._seg_dur += frame.samples_per_channel / frame.sample_rate
        if len(self._buf) >= int(SAMPLE_RATE * 2 * CHUNK_S):
            chunk = bytes(self._buf)
            self._buf.clear()
            asyncio.create_task(self._push(chunk))

    def flush(self) -> None:
        super().flush()
        if self._buf:
            chunk = bytes(self._buf)
            self._buf.clear()
            asyncio.create_task(self._push(chunk))
        # report playback-finished on the NOMINAL duration (realtime sim) so
        # the session's turn/pacing logic keeps a sane clock
        seg_dur, self._seg_dur = self._seg_dur, 0.0
        started = self._seg_started or time.time()
        remaining = max(0.0, started + seg_dur - time.time())

        async def _finish() -> None:
            await asyncio.sleep(remaining)
            self.on_playback_finished(playback_position=seg_dur, interrupted=False)
        if self._finish_task and not self._finish_task.done():
            self._finish_task.cancel()
        self._finish_task = asyncio.create_task(_finish())

    def clear_buffer(self) -> None:
        # barge-in: drop what's unsent and cut the engine's mouth
        self._buf.clear()
        seg_dur, self._seg_dur = self._seg_dur, 0.0
        if self._finish_task and not self._finish_task.done():
            self._finish_task.cancel()
        self.on_playback_finished(playback_position=seg_dur, interrupted=True)

        async def _cut() -> None:
            try:
                async with self._session().post(
                    f"{PUSH_URL}/interrupt_voice?sid={POD_SID}&persona=dance",
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as r:
                    logger.info(f"[POD-VOICE] barge-in interrupt → {r.status}")
            except Exception as e:
                logger.warning(f"[POD-VOICE] interrupt failed: {e}")
            self._state._pod_audible_until = time.time() + 0.5
        asyncio.create_task(_cut())
