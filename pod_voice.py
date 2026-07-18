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
import json
import time
import asyncio
import logging

import aiohttp
from livekit import rtc
from livekit.agents.voice import io as voice_io

logger = logging.getLogger("nova-pod-voice")

PUSH_URL = os.getenv("NOVA_POD_VOICE_URL",
                     "https://4mdjspk5xg3isf-8765.proxy.runpod.net")   # MuseTalk pod (2026-07-18)

# DIRECT ENGINE PUSH (2026-07-18 bridge-bypass): after the pod was dedicated to
# MuseTalk the relay bridge went stale — /push_voice answered ok:false and the
# FLV stream stayed digitally silent (-91dB) while a direct /humanaudio push
# was audible (-20dB). The worker now does what the bridge did: resample 24k→16k,
# wrap WAV, multipart to the engine. NOVA_POD_DIRECT=0 restores the bridge path.
ENGINE_URL = os.getenv("NOVA_POD_ENGINE_URL",
                       "https://4mdjspk5xg3isf-8011.proxy.runpod.net").rstrip("/")
DIRECT = os.getenv("NOVA_POD_DIRECT", "1") == "1"
LT_RATE = 16000   # MANDATORY — the engine's lip-sync conv crashes on other rates

try:
    import audioop   # stdlib until 3.12; removed in 3.13
except ImportError:  # pragma: no cover
    audioop = None

# one shared speech clock, same semantics the bridge kept (owner lock already
# guarantees a single pusher, so module-level is safe)
PUSH_CLOCK = {"speech_end": 0.0}


def _wav16k(pcm: bytes) -> bytes:
    import struct
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, LT_RATE, LT_RATE * 2, 2, 16) +
            b"data" + struct.pack("<I", len(pcm)) + pcm)

# OWNER LOCK registry — one avatar mouth, one owner room (newest wins).
POD_OWNER = {"room": None, "at": 0.0}


def claim_pod(room_name: str) -> None:
    POD_OWNER["room"] = room_name
    POD_OWNER["at"] = time.time()
    logging.getLogger("nova-pod-voice").info(f"[POD-VOICE] room '{room_name}' OWNS the avatar now")
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
        # OWNER LOCK (2026-07-15, "gibberish" fix): the pod avatar is ONE shared
        # mouth — every concurrent room used to push into the same stream and the
        # voices MIXED. Newest room claims ownership; a session that lost it
        # (zombie room, pre-refresh job) silently drops its audio.
        self._room_name = getattr(getattr(state, "room", None), "name", None) or f"state-{id(state)}"
        claim_pod(self._room_name)
        self._lost_logged = False
        self._rate_state = None   # audioop.ratecv carry between chunks

    def _to_16k(self, pcm: bytes) -> bytes:
        if audioop is not None:
            out, self._rate_state = audioop.ratecv(
                pcm, 2, 1, SAMPLE_RATE, LT_RATE, self._rate_state)
            return out
        # pure-python 24k→16k linear interp (audioop gone in 3.13)
        import array
        s = array.array("h")
        s.frombytes(pcm)
        n_out = len(s) * LT_RATE // SAMPLE_RATE
        if n_out <= 0 or len(s) < 2:
            return b""
        out = array.array("h", bytes(2 * n_out))
        step = (len(s) - 1) / max(1, n_out - 1)
        for i in range(n_out):
            pos = i * step
            j = int(pos)
            frac = pos - j
            nxt = s[j + 1] if j + 1 < len(s) else s[j]
            out[i] = int(s[j] * (1.0 - frac) + nxt * frac)
        return out.tobytes()

    def _session(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession()
        return self._http

    async def _push(self, pcm: bytes) -> None:
        if not pcm:
            return
        if POD_OWNER["room"] != self._room_name:
            if not self._lost_logged:
                self._lost_logged = True
                logger.warning(f"[POD-VOICE] ownership lost (owner={POD_OWNER['room']}) — "
                               f"room {self._room_name} stops pushing (zombie-mix guard)")
            return
        dur = len(pcm) / 2 / SAMPLE_RATE
        # MAGIC REVEAL SYNC (2026-07-15): tell the page her voice is REALLY flowing —
        # the reveal fade waits for first audio, and pod audio was invisible to it
        # (the page watched room-audio only → every pod reveal hit the 30s cap).
        _now = time.time()
        if _now - getattr(self, "_aud_announced", 0.0) > 1.5:
            self._aud_announced = _now
            _room = getattr(self._state, "room", None)
            if _room is not None:
                try:
                    asyncio.create_task(_room.local_participant.publish_data(
                        json.dumps({"kind": "pod-audible"}).encode("utf-8"), reliable=False))
                except Exception:
                    pass
        try:
            async with self._push_lock:   # chunks must land in order
                if DIRECT:
                    wav = _wav16k(self._to_16k(pcm))
                    form = aiohttp.FormData()
                    form.add_field("file", wav, filename="app.wav",
                                   content_type="audio/wav")
                    form.add_field("sessionid", str(POD_SID))  # STRING — engine crashes on int
                    async with self._session().post(
                        f"{ENGINE_URL}/humanaudio", data=form,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as r:
                        now = time.time()
                        PUSH_CLOCK["speech_end"] = max(PUSH_CLOCK["speech_end"], now) + dur
                        remain = PUSH_CLOCK["speech_end"] - now
                        self._state._pod_audible_until = (
                            now + remain + FLV_DELAY_S + ECHO_MARGIN_S)
                        logger.info(f"[POD-VOICE] direct-pushed {dur:.2f}s → {r.status} "
                                    f"(remain {remain:.2f}s, audible +{remain + FLV_DELAY_S:.1f}s)")
                else:
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
        # READY GATE (2026-07-15): no pod audio before the page's reveal —
        # "she starts gibberish before the page is ready". Early frames drop.
        try:
            if not self._state.client_ready.is_set():
                return
        except Exception:
            pass
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
            if POD_OWNER["room"] != self._room_name:
                return   # a zombie room must not cut the owner's speech
            try:
                if DIRECT:
                    PUSH_CLOCK["speech_end"] = 0.0
                    async with self._session().post(
                        f"{ENGINE_URL}/interrupt_talk",
                        json={"sessionid": POD_SID},
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as r:
                        logger.info(f"[POD-VOICE] barge-in interrupt (direct) → {r.status}")
                else:
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
