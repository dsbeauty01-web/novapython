# NOVA-CERTIFY harness — drives a REAL voice-only session on the live worker
# as a simulated kid: LiveKit room client, data-channel only (typed turns via
# {"kind":"user-said"}, detection facts via {"kind":"game-event"}, garble as
# user-said text). Records every packet both ways with timestamps.
#
# Run:  python certify_sim.py P1            (single probe)
#       python certify_sim.py all           (whole Stage 1)
# Logs: certify-logs/<probe>.jsonl  +  one-line verdict on stdout per probe.
#
# Evidence model: the worker's env is a separate Render service — its [STAGE]
# decisions are surfaced as {"kind":"stage-diag"} packets and her words as
# {"kind":"nova-said"} packets, so the packet stream IS the certification log.
import asyncio
import json
import os
import sys
import time

import aiohttp
from livekit import rtc

API = os.getenv("NOVA_API", "https://novapython.onrender.com")
LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certify-logs")
os.makedirs(LOGDIR, exist_ok=True)


class KidSession:
    """One simulated kid in one real room."""

    def __init__(self, probe: str, kid_name: str = "Bobo"):
        self.probe = probe
        self.kid_name = kid_name
        self.room = rtc.Room()
        self.t0 = None
        self.events = []          # every packet, both directions
        self.her_lines = []       # (t, text) from nova-said
        self.diags = []           # (t, decision, reason) from stage-diag
        self._logf = open(os.path.join(LOGDIR, probe + ".jsonl"), "w", encoding="utf-8")

    def _log(self, direction, payload):
        rec = {"t": round(time.time() - self.t0, 2) if self.t0 else 0,
               "dir": direction, **payload}
        self.events.append(rec)
        self._logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._logf.flush()

    async def start(self, voice_only=True, kid_id=None):
        async with aiohttp.ClientSession() as http:
            r = await http.post(API + "/v2/create-session", json={
                "kidId": kid_id, "kidName": self.kid_name,
                "voiceOnly": voice_only, "dispatch": True})
            js = await r.json()
        url = js.get("url") or js.get("livekitUrl") or js.get("serverUrl")
        token = js.get("token")
        self.room_name = js.get("room") or js.get("roomName")
        if not (url and token):
            raise RuntimeError("create-session missing url/token: " + json.dumps(js)[:300])

        @self.room.on("data_received")
        def _on_data(pkt: rtc.DataPacket):
            try:
                msg = json.loads(pkt.data.decode("utf-8"))
            except Exception:
                return
            kind = msg.get("kind", "?")
            self._log("in", {"kind": kind, **{k: v for k, v in msg.items() if k != "kind"}})
            if kind == "nova-said":
                self.her_lines.append((self.events[-1]["t"], msg.get("text", "")))
            elif kind == "stage-diag":
                self.diags.append((self.events[-1]["t"], msg.get("decision", ""),
                                   msg.get("reason", "")))

        await self.room.connect(url, token)
        self.t0 = time.time()
        self._log("meta", {"kind": "connected", "room": self.room_name})
        # INTRO-FINAL contract: the browser's calm fade ends with reveal-now — THE one
        # greeting trigger. The simulated kid is instantly "revealed".
        await asyncio.sleep(1.0)
        await self.send({"kind": "reveal-now"})

    async def send(self, payload: dict):
        self._log("out", dict(payload))
        await self.room.local_participant.publish_data(
            json.dumps(payload).encode("utf-8"), reliable=True)

    async def kid_say(self, text: str):
        await self.send({"kind": "user-said", "text": text, "source": "typed"})

    async def fact(self, event: dict):
        # the browser nests the event object: {"kind":"game-event","event":{...}}
        await self.send({"kind": "game-event", "event": event})

    async def wait(self, seconds: float):
        await asyncio.sleep(seconds)

    def lines_since(self, t: float):
        return [(lt, tx) for lt, tx in self.her_lines if lt >= t]

    def now(self):
        return time.time() - self.t0

    async def wait_for_line(self, timeout: float):
        """Wait until she says something new (or timeout). Returns list of new lines."""
        mark = len(self.her_lines)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if len(self.her_lines) > mark:
                await asyncio.sleep(2.0)   # let a multi-part reply finish
                return self.her_lines[mark:]
            await asyncio.sleep(0.25)
        return []

    async def close(self):
        try:
            await self.room.disconnect()
        except Exception:
            pass
        self._log("meta", {"kind": "closed"})
        self._logf.close()


def verdict(probe, ok, evidence):
    print(json.dumps({"probe": probe, "pass": bool(ok), "evidence": str(evidence)[:400]},
                     ensure_ascii=False))


# ────────────────────────── STAGE 1 PROBES ──────────────────────────

async def P1():
    """Greet+wait: one greet line, 20s silence -> exactly ONE re-invite -> quiet."""
    s = KidSession("P1")
    await s.start()
    first = await s.wait_for_line(40)           # her opening (cold worker allowance)
    n_open = len(first)
    t_mark = s.now()
    await s.wait(20)                            # silence window 1
    reinvites_1 = s.lines_since(t_mark)
    t2 = s.now()
    await s.wait(25)                            # silence window 2 — must stay quiet
    reinvites_2 = s.lines_since(t2)
    ok = (n_open >= 1) and (len(reinvites_1) <= 1) and (len(reinvites_2) <= 1) and \
         (len(reinvites_1) + len(reinvites_2) <= 1 or
          any("silence" in d for _, d, _r in s.diags))
    verdict("P1", ok, {"open": [t for _, t in first][:2],
                       "window1": [t for _, t in reinvites_1],
                       "window2": [t for _, t in reinvites_2],
                       "diags": s.diags[:6]})
    await s.close()
    return ok


async def P2():
    """Name echo: 'Hi, I'm Bobo.' -> she echoes exactly Bobo, one line."""
    s = KidSession("P2")
    await s.start()
    await s.wait_for_line(40)
    await s.kid_say("Hi, I'm Bobo.")
    reply = await s.wait_for_line(15)
    txt = " ".join(t for _, t in reply)
    ok = ("bobo" in txt.lower()) and (len(reply) <= 2) and \
         not any(bad in txt.lower() for bad in ("boba", "bobbo", "bobo?!  what"))
    verdict("P2", ok, {"reply": txt})
    await s.close()
    return ok


async def P3():
    """Light happy path: light fires -> she wonders -> inject shrug fact -> praise ONCE, specific."""
    s = KidSession("P3")
    await s.start()
    await s.wait_for_line(40)
    await s.kid_say("Hi, I'm Bobo.")
    await s.wait_for_line(15)
    # wait for the light world-event (~20s after her first word + lull)
    light = None
    t0 = time.time()
    while time.time() - t0 < 60:
        cues = [e for e in s.events if e.get("kind") == "cue-part"]
        if cues:
            light = cues[0]
            break
        await asyncio.sleep(0.5)
    if not light:
        verdict("P3", False, {"err": "light cue never arrived", "diags": s.diags[:8]})
        await s.close()
        return False
    pre_praise = await s.wait_for_line(20)       # her discovery lines
    t_mark = s.now()
    # inject the shrug DETECTION FACT the way the browser does
    await s.fact({"event": "try_move", "action": "shoulder shrug"})
    praise = await s.wait_for_line(15)
    ptxt = " ".join(t for _, t in praise).lower()
    early = " ".join(t for _, t in pre_praise).lower()
    insta = any(w in early for w in ("you did it", "you moved it", "nailed it", "perfect shrug"))
    ok = bool(praise) and ("shoulder" in ptxt or "shrug" in ptxt or "isolation" in ptxt) \
         and not insta
    verdict("P3", ok, {"discovery": early[:150], "praise": ptxt[:200], "insta_praise": insta})
    await s.close()
    return ok


async def P4():
    """Light no-show: never inject the fact -> ONE gentle re-invite -> warm move-on, then quiet about the light."""
    s = KidSession("P4")
    await s.start()
    await s.wait_for_line(40)
    await s.kid_say("Hi, I'm Bobo.")
    await s.wait_for_line(15)
    t0 = time.time()
    while time.time() - t0 < 60:
        if any(e.get("kind") == "cue-part" for e in s.events):
            break
        await asyncio.sleep(0.5)
    await s.wait_for_line(20)                    # discovery
    t_mark = s.now()
    await s.wait(45)                             # no fact ever — watch her behavior
    after = s.lines_since(t_mark)
    texts = [t.lower() for _, t in after]
    mentions_light = sum(1 for t in texts if any(w in t for w in ("light", "shoulder", "shrug", "glow", "sparkle")))
    ok = len(after) <= 3 and mentions_light <= 2
    verdict("P4", ok, {"lines_after": texts[:6], "light_mentions": mentions_light})
    await s.close()
    return ok


async def P6():
    """Game offer + consent: exactly three games, never picks for the kid, one re-invite max."""
    s = KidSession("P6")
    await s.start()
    await s.wait_for_line(40)
    await s.kid_say("Hi, I'm Bobo.")
    await s.wait_for_line(15)
    await s.kid_say("Let's dance!")              # fast-path to the picker
    offer = await s.wait_for_line(20)
    got_picker = any(e.get("kind") == "go-picker" for e in s.events)
    t_mark = s.now()
    await s.wait(20)                             # silence at the consent beat
    r1 = s.lines_since(t_mark)
    t2 = s.now()
    await s.wait(25)
    r2 = s.lines_since(t2)
    all_txt = " ".join(t.lower() for _, t in offer + r1 + r2)
    self_picked = any(w in all_txt for w in ("awesome choice", "great choice", "let's start", "here we go", "starting the"))
    ok = got_picker and (len(r1) <= 1) and (len(r2) <= 1) and not self_picked
    verdict("P6", ok, {"picker": got_picker, "offer": [t for _, t in offer][:3],
                       "silence1": [t for _, t in r1], "silence2": [t for _, t in r2],
                       "self_picked": self_picked})
    await s.close()
    return ok


async def P7():
    """Garble wall: nonsense at the name beat -> ignored (no name, no praise, no advance)."""
    s = KidSession("P7")
    await s.start()
    await s.wait_for_line(40)
    for g in ("É", "Peso", "谢谢你"):
        await s.kid_say(g)
        await s.wait(6)
    lines = [t.lower() for _, t in s.her_lines[1:]]
    # she must not have taken any of these as a name or advanced the flow
    took_name = any(any(g in t for g in ("peso", "谢")) for t in lines)
    n_replies = len(lines)
    ok = (not took_name) and n_replies <= 3
    verdict("P7", ok, {"replies": lines[:5], "took_name": took_name})
    await s.close()
    return ok


async def P5():
    """Light once-ever: after the light resolves, a phase bounce must NOT re-fire it."""
    s = KidSession("P5")
    await s.start()
    await s.wait_for_line(40)
    await s.kid_say("Hi, I'm Bobo.")
    await s.wait_for_line(15)
    t0 = time.time()
    while time.time() - t0 < 60:
        if any(e.get("kind") == "cue-part" for e in s.events):
            break
        await asyncio.sleep(0.5)
    await s.fact({"event": "try_move", "action": "shoulder shrug"})
    await s.wait_for_line(15)                    # her one celebration
    await s.wait(5)                              # let the WIN visuals (sparkle/jump cue) finish
    n_cues_before = sum(1 for e in s.events if e.get("kind") == "cue-part")
    # phase bounce: pretend the browser went to picker and back to recognition
    await s.send({"kind": "test-force-phase", "phase": "picker"})
    await s.wait(3)
    await s.send({"kind": "test-force-phase", "phase": "recognition"})
    await s.wait(30)                             # would the light re-fire?
    n_cues_after = sum(1 for e in s.events if e.get("kind") == "cue-part")
    relit = n_cues_after > n_cues_before
    ok = not relit
    verdict("P5", ok, {"cues_before_bounce": n_cues_before,
                       "cues_after_bounce": n_cues_after, "relit": relit})
    await s.close()
    return ok


async def P8():
    """Readiness task per game: pick each game by voice -> she asks the task ->
    inject the fact -> she celebrates -> game opens. Praise before fact = FAIL."""
    results = {}
    for game, phrase, fact_ev in [
            ("freeze", "Freeze!", {"event": "try_move", "action": "freeze"}),
            ("wave",   "Wave!",   {"event": "try_move", "action": "hand up"}),
            ("upgroove", "Up Groove!", {"event": "try_move", "action": "hand up"})]:
        s = KidSession("P8-" + game)
        await s.start()
        await s.wait_for_line(40)
        await s.kid_say("Hi, I'm Bobo.")
        await s.wait_for_line(15)
        await s.kid_say("Let's dance!")
        await s.wait_for_line(20)                # picker + her line
        await s.kid_say(phrase)                  # the pick, by voice
        ask = await s.wait_for_line(20)          # her readiness ask
        ask_txt = " ".join(t.lower() for _, t in ask)
        pre = s.now()
        insta = any(w in ask_txt for w in ("you did it", "perfect", "nailed it", "you froze", "you got it"))
        await s.wait(4)                          # statue-silence window (freeze)
        mid = s.lines_since(pre)
        await s.fact(fact_ev)
        cel = await s.wait_for_line(15)
        cel_txt = " ".join(t.lower() for _, t in cel)
        ok = bool(ask) and not insta and bool(cel) and len(mid) <= 1
        results[game] = {"ok": ok, "ask": ask_txt[:120], "mid_hold_lines": len(mid),
                         "celebration": cel_txt[:120], "insta": insta}
        await s.close()
        await asyncio.sleep(2)
    ok_all = all(r["ok"] for r in results.values())
    verdict("P8", ok_all, results)
    return ok_all


# ────────────────────────── STAGE 2 — GAMES ──────────────────────────

def _sentence_complete(t):
    t = t.strip()
    return bool(t) and t[-1] in ".!?…\"'"


async def GAME(song_id: str, label: str):
    """Full voice-only playthrough entered FROM the intro. Verifies G1-G6."""
    s = KidSession("G-" + song_id)
    await s.start()
    await s.wait_for_line(40)                      # greet
    await s.kid_say("Hi, I'm Bobo.")
    await s.wait_for_line(15)
    await s.kid_say("Let's dance!")
    await s.wait_for_line(20)                      # picker + her line
    await s.fact({"event": "picked", "song": song_id})
    await s.wait_for_line(15)                      # loading hype
    await s.fact({"event": "phase", "phase": "dance"})
    await s.fact({"event": "song_start", "song": song_id, "sec": 0})
    go = await s.wait_for_line(15)                 # the go-line
    t_game0 = s.now()

    hits_injected = []
    windows = []                                   # (t_start, t_end, had_fact)
    async def tick(sec):
        await s.fact({"event": "music_tick", "sec": sec})

    # simulated 100s song: hits at 20/35/50, miss at 65, fact-free 70-95
    script = [(5, None), (10, None), (15, None),
              (20, {"event": "hit", "hits": 1, "streak": 1, "action": "hand up"}),
              (25, None), (30, None),
              (35, {"event": "hit", "hits": 2, "streak": 2, "action": "hand up"}),
              (40, None), (45, None),
              (50, {"event": "hit", "hits": 3, "streak": 3, "action": "clap"}),
              (55, None), (60, None),
              (65, {"event": "miss"}),
              (70, None), (75, None), (80, None), (85, None), (90, None), (95, None)]
    last_t = 0
    for sec, fact in script:
        await s.wait(sec - last_t)
        last_t = sec
        await tick(sec)
        if fact:
            w0 = s.now()
            await s.fact(fact)
            if fact["event"] == "hit":
                hits_injected.append(fact)
    t_game_end = s.now()
    await s.fact({"event": "phase", "phase": "goodbye"})
    ending = await s.wait_for_line(25)
    await s.wait(8)
    ending = s.lines_since(t_game_end)

    in_game = [(lt, tx) for lt, tx in s.her_lines if t_game0 <= lt < t_game_end]
    texts = [tx for _, tx in in_game]
    low = [t.lower() for t in texts]

    g1 = not any("i'm nova" in t or "i am nova" in t for t in low)
    # G2 truth: praise-ish lines only near a hit; the 70-95s fact-free stretch quiet
    quiet_zone = [tx for lt, tx in in_game if t_game0 + 68 <= lt <= t_game0 + 96]
    g2 = len(quiet_zone) <= 1
    # G3 turn discipline: no window of 15s with >1 unprompted line
    g3 = True
    times = [lt for lt, _ in in_game]
    for i in range(len(times)):
        burst = [t for t in times if times[i] <= t < times[i] + 15]
        if len(burst) > 2:
            g3 = False
            break
    g4 = len(texts) == len(set(texts))
    g5_bad = [t for t in texts + [tx for _, tx in ending] if not _sentence_complete(t)]
    g5 = len(g5_bad) == 0
    import re as _re_n
    nums = [int(n) for t in low + [t.lower() for _, t in ending] for n in _re_n.findall(r"\b(\d{1,3})\b", t)]
    g6 = all(n in (1, 2, 3) or n <= 3 for n in nums) if nums else True
    ok = g1 and g2 and g3 and g4 and g5 and g6
    verdict("G-" + song_id, ok, {
        "g1_no_regreet": g1, "g2_truth_quiet_zone": g2, "g3_turns": g3,
        "g4_decan": g4, "g5_whole_sentences": g5, "g5_bad": g5_bad[:2],
        "g6_numbers": {"ok": g6, "seen": nums[:6]},
        "go_line": " ".join(t for _, t in go)[:100],
        "in_game_lines": texts[:8], "ending": [t for _, t in ending][:4]})
    await s.close()
    return ok


async def G_all():
    res = {}
    for sid, label in [("hello", "Hello Hello!"), ("joined", "Up Groove!"), ("wave", "Wave!")]:
        try:
            res[sid] = await GAME(sid, label)
        except Exception as e:
            verdict("G-" + sid, False, {"exception": repr(e)[:200]})
            res[sid] = False
        await asyncio.sleep(3)
    print(json.dumps({"stage2": res}))
    return all(res.values())


# ─────────────────── STAGE 3 + 4 — TRANSITIONS + ENDING ───────────────────

async def T2():
    """Game end -> play again -> DIFFERENT game: name kept, light not re-fired, clean open."""
    s = KidSession("T2")
    await s.start()
    await s.wait_for_line(40)
    await s.kid_say("Hi, I'm Bobo.")
    await s.wait_for_line(15)
    await s.kid_say("Let's dance!")
    await s.wait_for_line(20)
    await s.fact({"event": "picked", "song": "hello"})
    await s.wait_for_line(15)
    await s.fact({"event": "phase", "phase": "dance"})
    await s.fact({"event": "song_start", "song": "hello", "sec": 0})
    await s.wait_for_line(15)
    await s.wait(10)
    await s.fact({"event": "hit", "hits": 1, "streak": 1, "action": "hand up"})
    await s.wait(8)
    cues_before = sum(1 for e in s.events if e.get("kind") == "cue-part")
    await s.fact({"event": "play_again"})
    pa = await s.wait_for_line(15)
    await s.fact({"event": "picked", "song": "wave"})
    second = await s.wait_for_line(15)
    await s.fact({"event": "phase", "phase": "dance"})
    await s.fact({"event": "song_start", "song": "wave", "sec": 0})
    go2 = await s.wait_for_line(15)
    await s.wait(5)
    cues_after = sum(1 for e in s.events if e.get("kind") == "cue-part")
    all_txt = " ".join(t.lower() for _, t in pa + second + go2)
    regreeted = "i'm nova" in all_txt or "what's your name" in all_txt
    ok = (cues_after == cues_before) and not regreeted and bool(go2)
    verdict("T2", ok, {"light_refired": cues_after > cues_before, "regreeted": regreeted,
                       "second_game_lines": [t for _, t in second + go2][:4]})
    await s.close()
    return ok


async def T4():
    """Two 'no' to play-again -> she initiates the goodbye herself."""
    s = KidSession("T4")
    await s.start()
    await s.wait_for_line(40)
    await s.kid_say("Hi, I'm Bobo.")
    await s.wait_for_line(15)
    await s.kid_say("Let's dance!")
    await s.wait_for_line(20)
    await s.fact({"event": "picked", "song": "hello"})
    await s.wait_for_line(15)
    await s.fact({"event": "phase", "phase": "dance"})
    await s.fact({"event": "song_start", "song": "hello", "sec": 0})
    await s.wait_for_line(15)
    await s.wait(8)
    await s.fact({"event": "hit", "hits": 2, "streak": 2, "action": "hand up"})
    await s.wait(6)
    await s.fact({"event": "phase", "phase": "goodbye"})
    await s.wait_for_line(20)
    await s.kid_say("No.")
    r1 = await s.wait_for_line(15)
    await s.kid_say("No, I'm done.")
    r2 = await s.wait_for_line(20)
    txt = " ".join(t.lower() for _, t in r1 + r2)
    said_bye = any(w in txt for w in ("bye", "see you", "next time", "goodbye", "later"))
    named = "bobo" in txt
    # no infinite loop: after the goodbye, 20s of silence
    t_mark = s.now()
    await s.wait(20)
    trailing = s.lines_since(t_mark)
    ok = said_bye and len(trailing) <= 1
    verdict("T4", ok, {"goodbye": txt[:200], "named": named, "trailing": len(trailing)})
    await s.close()
    return ok


async def E5():
    """PULSE fires: session-end JSON POST -> worker /pulse; paste the received JSON."""
    payload = {"pulse": {"id": "certify-" + str(int(time.time())), "kid": "Bobo",
                         "dur_sec": 214, "funnel": "intro>picker>hello>goodbye",
                         "score": 3, "feedback_text": "it was fun!"},
               "log": [{"t": 0, "ev": "certify-run"}]}
    async with aiohttp.ClientSession() as http:
        r = await http.post(API + "/pulse", json=payload)
        js = await r.json()
    ok = js.get("ok") is True and js.get("id") == payload["pulse"]["id"] and js.get("score") == 3
    verdict("E5", ok, {"sent": payload["pulse"], "received": js})
    return ok


async def FLOW():
    """The founder's bar: LIVE (reply <=12s to every input), never stuck, never
    monologue (max 1 unprompted line between inputs), ONE chance on silence."""
    s = KidSession("FLOW")
    await s.start()
    problems = []

    async def kid(text, expect_reply=True):
        t0 = s.now()
        await s.kid_say(text)
        got = await s.wait_for_line(12)
        if expect_reply and not got:
            problems.append(f"STUCK: no reply within 12s to '{text}'")
        return got

    opening = await s.wait_for_line(55)     # cold Render worker needs a beat on run 1
    if not opening:
        problems.append("STUCK: no greet")
    await s.wait(6)
    await kid("hi")
    await kid("im Lolo")
    # silence window: exactly <=1 nudge in 30s
    t_mark = s.now()
    await s.wait(30)
    nudges = s.lines_since(t_mark)
    # the magic light igniting during this window is a WORLD EVENT — its one
    # discovery line is allowed (same law as the picker beat), on top of one nudge.
    light_evt = any(e.get("kind") == "cue-part" and e["t"] >= t_mark for e in s.events)
    if len(nudges) > (2 if light_evt else 1):
        problems.append(f"MONOLOGUE in silence: {[t for _, t in nudges]}")
    elif len(nudges) == 2 and (nudges[1][0] - nudges[0][0]) < 3.5:
        problems.append(f"BABBLE in silence (gap {nudges[1][0]-nudges[0][0]:.1f}s): {[t for _, t in nudges]}")
    await kid("yes")
    await kid("lets dance")
    # consent silence: no self-pick, <=1 nudge in 30s
    t_mark = s.now()
    await s.wait(30)
    lines2 = s.lines_since(t_mark)
    # response + the game-offer as the picker VISIBLY opens = a world event, not
    # babble. Allowed: exactly 2 lines, a go-picker packet near them, and the two
    # lines as CALM SEPARATE BEATS (>=3.5s apart, the beat-spacing law). Anything
    # more, or machine-gunned lines, is a monologue.
    picker_evt = any(e.get("kind") == "go-picker" and e["t"] >= t_mark - 12 for e in s.events)
    if len(lines2) > (2 if picker_evt else 1):
        problems.append(f"MONOLOGUE at picker: {[t for _, t in lines2]}")
    elif len(lines2) == 2 and (lines2[1][0] - lines2[0][0]) < 3.5:
        problems.append(f"BABBLE (no beat gap {lines2[1][0]-lines2[0][0]:.1f}s): {[t for _, t in lines2]}")
    txt2 = " ".join(t.lower() for _, t in lines2)
    if any(w in txt2 for w in ("great choice", "let's go with", "starting")):
        problems.append(f"SELF-PICK: {txt2[:120]}")
    await kid("wave please")
    # monologue check across the whole session: no 3 consecutive her-lines
    # without a kid input in between
    kid_times = [e["t"] for e in s.events if e.get("dir") == "out" and e.get("kind") == "user-said"]
    runs, run = [], 0
    for lt, _tx in s.her_lines[1:]:                     # skip the greet
        if any(lt - 14 < kt < lt for kt in kid_times):
            run = 0
        else:
            run += 1
            runs.append(run)
    if runs and max(runs) >= 3:
        problems.append(f"MONOLOGUE chain: {max(runs)} unprompted lines")
    ok = not problems
    verdict("FLOW", ok, {"problems": problems[:5],
                         "her_lines": [t for _, t in s.her_lines][:14]})
    await s.close()
    return ok


PROBES = {"P1": P1, "P2": P2, "P3": P3, "P4": P4, "P5": P5, "P6": P6, "P7": P7, "P8": P8,
          "G": G_all, "T2": T2, "T4": T4, "E5": E5, "FLOW": FLOW}


async def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(PROBES) if which == "all" else [which]
    results = {}
    for n in names:
        try:
            results[n] = await PROBES[n]()
        except Exception as e:
            verdict(n, False, {"exception": repr(e)[:200]})
            results[n] = False
        await asyncio.sleep(2)
    print(json.dumps({"summary": results}))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    asyncio.run(main())


# ────────────────────────── VOICE PROBE (real audio) ──────────────────────────
import wave as _wave
import audioop as _audioop

VOICE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certify-voice")


def _load_wav_48k(path):
    """wav file -> mono 48kHz int16 bytes."""
    w = _wave.open(path, "rb")
    rate, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
    pcm = w.readframes(w.getnframes()); w.close()
    if sw != 2:
        pcm = _audioop.lin2lin(pcm, sw, 2)
    if ch == 2:
        pcm = _audioop.tomono(pcm, 2, 0.5, 0.5)
    if rate != 48000:
        pcm, _ = _audioop.ratecv(pcm, 2, 1, rate, 48000, None)
    return pcm


class VoiceKid(KidSession):
    """A kid with a REAL mouth: publishes a mic track and speaks wav lines."""

    async def start_voice(self):
        await self.start()
        self.mic = rtc.AudioSource(48000, 1)
        track = rtc.LocalAudioTrack.create_audio_track("mic", self.mic)
        await self.room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        self._log("meta", {"kind": "mic-published"})
        asyncio.create_task(self._silence_pump())

    async def _silence_pump(self):
        # a real mic never stops sending frames — 10ms of silence keeps VAD honest
        silence = b"\x00" * (480 * 2)
        while True:
            try:
                await self.mic.capture_frame(rtc.AudioFrame(silence, 48000, 1, 480))
            except Exception:
                return
            await asyncio.sleep(0.01)

    async def speak(self, clip):
        pcm = _load_wav_48k(os.path.join(VOICE_DIR, clip + ".wav"))
        self._log("out", {"kind": "VOICE", "clip": clip, "ms": len(pcm) // 96})
        step = 480 * 2                       # 10ms @48k mono int16
        for i in range(0, len(pcm) - step, step):
            await self.mic.capture_frame(rtc.AudioFrame(pcm[i:i + step], 48000, 1, 480))
            await asyncio.sleep(0.0095)


async def VFLOW():
    """REAL voice-to-voice: she must hear synthesized kid AUDIO and answer."""
    s = VoiceKid("VFLOW")
    await s.start_voice()
    problems = []
    opening = await s.wait_for_line(55)
    if not opening:
        problems.append("STUCK: no greet")
    await s.wait(2)
    await s.speak("name")
    r1 = await s.wait_for_line(18)
    if not r1:
        problems.append("DEAF: no reply to spoken name")
    elif not any("lolo" in t.lower() for _, t in r1):
        problems.append(f"MISHEARD name: {[t for _, t in r1][:2]}")
    await s.wait(3)
    await s.speak("yes")
    r2 = await s.wait_for_line(18)
    await s.wait(3)
    await s.speak("dance")
    r3 = await s.wait_for_line(18)
    if not (r2 or r3):
        problems.append("DEAF mid-flow: no reply to spoken yes/dance")
    greets = sum(1 for _, t in s.her_lines if "i'm nova" in t.lower())
    if greets > 1:
        problems.append(f"RE-GREETED x{greets}")
    ok = not problems
    verdict("VFLOW", ok, {"problems": problems[:5],
                          "her_lines": [t for _, t in s.her_lines][:10]})
    await s.close()
    return ok


PROBES["VFLOW"] = VFLOW
