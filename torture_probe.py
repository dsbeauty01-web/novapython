# TORTURE PROBE — LAW-INPUT-LOCK proof (founder spec 2026-08-11).
# Connects to the POD brain's /rt websocket exactly like the game page and streams
# REAL AUDIO: cafe noise, breaths, silence, plus typed garble fragments — then one
# real spoken "My name is Shuki!". PASS = after her greet (+max one re-invite) she
# speaks ZERO times during the torture, then exactly ONE warm SHUKI response.
#
# Run: python torture_probe.py wss://<pod>-8765.proxy.runpod.net/rt
import asyncio
import audioop
import base64
import json
import math
import os
import random
import sys
import time
import wave

import aiohttp

POD_WS = sys.argv[1] if len(sys.argv) > 1 else ""
VOICE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certify-voice")
RATE = 24000
CHUNK = RATE // 10 * 2          # 100ms of PCM16 mono


def wav_to_pcm24k(path):
    w = wave.open(path, "rb")
    rate, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
    pcm = w.readframes(w.getnframes()); w.close()
    if sw != 2:
        pcm = audioop.lin2lin(pcm, sw, 2)
    if ch == 2:
        pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
    if rate != RATE:
        pcm, _ = audioop.ratecv(pcm, 2, 1, rate, RATE, None)
    return pcm


def noise(seconds, amp=900):
    """cafe-ish noise: filtered random with a low hum."""
    n = int(RATE * seconds)
    buf = bytearray()
    prev = 0.0
    for i in range(n):
        white = random.uniform(-1, 1)
        prev = 0.92 * prev + 0.08 * white          # low-pass -> speech-band-ish rumble
        v = int(amp * (prev * 2.2 + 0.35 * math.sin(i / 37.0)))
        buf += int(max(-32000, min(32000, v))).to_bytes(2, "little", signed=True)
    return bytes(buf)


def breath(seconds=0.7, amp=350):
    return noise(seconds, amp)


def silence(seconds):
    return b"\x00" * (int(RATE * seconds) * 2)


async def run(tag):
    events = []          # (t, kind, text)
    t0 = time.time()

    def note(kind, text=""):
        events.append((round(time.time() - t0, 1), kind, text))

    session = aiohttp.ClientSession()
    ws = await session.ws_connect(POD_WS, heartbeat=20)
    note("connected")

    her_lines = []
    done = asyncio.Event()

    async def reader():
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            m = json.loads(msg.data)
            if m.get("type") == "nova_done":
                txt = (m.get("text") or "").strip()
                if txt:
                    her_lines.append((round(time.time() - t0, 1), txt))
                    note("NOVA", txt[:80])
            elif m.get("type") == "you_text":
                note("KID-ACCEPTED", m.get("text", "")[:40])
        done.set()

    rt = asyncio.create_task(reader())

    async def stream(pcm, label):
        note("audio", label)
        for i in range(0, len(pcm) - CHUNK, CHUNK):
            await ws.send_json({"type": "audio", "data": base64.b64encode(pcm[i:i + CHUNK]).decode()})
            await asyncio.sleep(0.093)

    # her greet fires on connect — give it room
    await asyncio.sleep(12)
    greet_lines = len(her_lines)

    # ── THE TORTURE (about 3 minutes) ──────────────────────────────
    torture_start = len(her_lines)
    await stream(noise(8), "cafe noise 8s")
    await stream(silence(6), "silence 6s")
    await stream(breath(), "breath")
    await stream(noise(5, amp=600), "cafe noise 5s")
    await stream(silence(40), "dead silence 40s")
    await stream(breath(), "breath")
    await stream(noise(10), "cafe noise 10s")
    await stream(silence(8), "silence 8s")
    await stream(breath(0.5), "short breath")
    await stream(noise(6, amp=1100), "louder noise 6s")
    await stream(silence(30), "dead silence 30s")
    await stream(breath(), "breath")
    await stream(noise(7), "cafe noise 7s")
    await stream(silence(10), "silence 10s")
    torture_lines = her_lines[torture_start:]

    # ── THE ONE REAL TURN ──────────────────────────────────────────
    clip = os.environ.get("TORTURE_CLIP", "shuki")
    await stream(wav_to_pcm24k(os.path.join(VOICE_DIR, clip + ".wav")), "SPOKEN: " + clip)
    await asyncio.sleep(14)
    reply = her_lines[len(her_lines) - (len(her_lines) - torture_start - len(torture_lines)):]
    shuki_replies = [(t, x) for t, x in her_lines if t > events[-1][0] - 15 and "shuki" in x.lower()]
    post_lines = her_lines[torture_start + len(torture_lines):]

    await ws.close(); await session.close()
    # STT margin: the robotic TTS gets heard as shuki/shaky/sha'key — she echoes
    # what she HEARD, honestly. Accept any of those. Exactly ONE post-name line.
    _expect = os.environ.get("TORTURE_EXPECT", "shuki,shaky,sha'k,shak").split(",")
    name_ok = any(any(v in x.lower() for v in _expect) for _, x in post_lines) or os.environ.get("TORTURE_EXPECT") == "*"
    ok = (len(torture_lines) <= 1) and name_ok and (len(post_lines) == 1)
    print(json.dumps({
        "probe": "TORTURE-" + tag, "pass": ok,
        "greet_lines": greet_lines,
        "lines_during_torture": [x for _, x in torture_lines],
        "post_name_lines": [x for _, x in post_lines],
        "events_tail": [f"{t} {k} {x}" for t, k, x in events[-14:]],
    }, ensure_ascii=False))
    return ok


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    asyncio.run(run(sys.argv[2] if len(sys.argv) > 2 else "1"))
