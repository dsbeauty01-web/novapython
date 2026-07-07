# PHASE 3 SIM — machine-checks the REAL GameVoiceGate/router (personality.py)
# against the PHASE3-GAME.md self-test list. Driven by tools/phase3-loop.js
# (node) which spawns this with any Python 3. Pure stdlib, simulated clock —
# no livekit, no network. Prints one JSON blob on stdout.
import asyncio
import json
import os
import sys

try:  # Windows consoles default to a legacy codepage — the JSON is UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import personality as P

CHECKS = []


def check(tid, name, ok, evidence):
    CHECKS.append({"id": tid, "name": name, "pass": bool(ok), "evidence": str(evidence)[:300]})


# ──────────────────────────────────────────────────────────────────────
# song simulator: cues open, resolve to hit/miss per a pattern string.
# H=hit M=miss F=freeze-hit. Returns (gate, spoken, windows)
# spoken = [(t, key, milestone, action)] ; windows = [(open, close)]
# ──────────────────────────────────────────────────────────────────────
def run_song(length, pattern, cue_start=6.0, cue_every=3.0, resolve_after=1.2,
             moments=(), all_time_best=0, conf=None):
    g = P.GameVoiceGate(all_time_best=all_time_best)
    spoken, windows = [], []
    events = []                                   # (t, ev)
    for s in range(0, int(length) + 1):
        events.append((float(s), {"event": "music_tick", "sec": s}))
    streak = 0
    t = cue_start
    i = 0
    while t < length - 1.5 and i < len(pattern):
        c = pattern[i]
        events.append((t, {"event": "move_cue", "action": "hand-right" if c != "F" else "freeze"}))
        windows.append((t, t + resolve_after))
        if c == "M":
            streak = 0
            ev = {"event": "miss", "action": "hand-right"}
            if conf:
                ev["confidence"] = conf
        elif c == "F":
            ev = {"event": "freeze_hit"}
        else:
            streak += 1
            ev = {"event": "hit", "streak": streak, "action": "hand-right"}
        events.append((t + resolve_after, ev))
        t += cue_every
        i += 1
    for (mt, label) in moments:
        events.append((mt, {"event": "music_moment", "label": label}))
    events.sort(key=lambda x: x[0])
    BASE = 10000.0                                # wall-clock offset, arbitrary
    for (ts, ev) in events:
        d = g.decide(ev, BASE + ts)
        if d["action"] in ("premade", "live"):
            spoken.append((ts, d["key"], d["milestone"], d["action"]))
    return g, spoken, windows


# ══ TEST 1 — budget + cooldown (Hello Hello 111s, Wave 28s) ══════════
pat = "HHMHHHMHHMHHHHMHHMHHFHHMHHHHMHHMH"        # 33 cues, mixed kid, no 3-miss runs
g1, spoken1, windows1 = run_song(111, pat, moments=((30, "drop"), (62, "section"), (90, "drop")))
per_min = len(spoken1) / (111 / 60.0)
check(1, "budget: Hello Hello 111s beats/min ~4-6",
      2.5 <= per_min <= 7.0,
      f"{len(spoken1)} beats in 111s = {per_min:.1f}/min → {[(round(t,1),k) for t,k,m,a in spoken1]}")
gaps_bad = []
for a, b in zip(spoken1, spoken1[1:]):
    gap = b[0] - a[0]
    if gap < 2.5 - 1e-9 and not b[2]:             # non-milestone must respect cooldown
        gaps_bad.append((round(a[0], 1), round(b[0], 1), round(gap, 2)))
check(1, "cooldown >=2500ms between lines (milestones exempt)", not gaps_bad,
      gaps_bad or f"min non-milestone gap ok across {len(spoken1)} beats")
gw, spokenw, _ = run_song(28, "HHHHHHHH", cue_start=5.0)
check(1, "budget: Wave 28s ≈ 2-3 beats (≤4)", 1 <= len(spokenw) <= 4,
      f"{len(spokenw)} beats → {[(round(t,1),k) for t,k,m,a in spokenw]}")

# ══ TEST 2 — no line inside an open cue window ═══════════════════════
inside = [(round(t, 1), k) for (t, k, m, a) in spoken1
          for (o, c) in windows1 if o < t < c - 1e-9]
check(2, "no speech during an open cue window", not inside,
      inside or f"checked {len(spoken1)} beats against {len(windows1)} windows")

# ══ TEST 3 — miss rules ══════════════════════════════════════════════
g3 = P.GameVoiceGate()
g3.decide({"event": "music_tick", "sec": 30}, 100.0)  # past settle
d = g3.decide({"event": "miss", "confidence": "high"}, 103.0)
check(3, "single miss → zero voice", d["action"] == "silent", d["reason"])
d2 = g3.decide({"event": "miss", "confidence": "high"}, 106.0)
d3 = g3.decide({"event": "miss", "confidence": "high"}, 109.0)
d4 = g3.decide({"event": "miss", "confidence": "high"}, 112.0)
check(3, "3 misses + HIGH conf → exactly ONE live blurt",
      d2["action"] == "silent" and d3["action"] == "live" and d3["key"] == "blurt"
      and d4["action"] == "silent",
      f"m2={d2['action']} m3={d3['action']}/{d3['key']} m4={d4['action']}({d4['reason']})")
g3b = P.GameVoiceGate()
g3b.decide({"event": "music_tick", "sec": 30}, 100.0)
acts = [g3b.decide({"event": "miss"}, 100.0 + 3 * i)["action"] for i in range(1, 5)]
check(3, "3 misses + LOW/no conf → silence (guard)", all(a == "silent" for a in acts),
      f"actions={acts} last_reason={g3b.log[-1]['reason']}")

# ══ TEST 4 — streak 3 / 5 escalate ═══════════════════════════════════
g4 = P.GameVoiceGate(all_time_best=10)               # no newbest noise
g4.decide({"event": "music_tick", "sec": 30}, 100.0)
keys4 = []
for i, s in enumerate([1, 2, 3, 4, 5]):
    d = g4.decide({"event": "hit", "streak": s, "action": "hand-right"}, 103.0 + i * 3)
    keys4.append((s, d["action"], d["key"]))
s3 = [k for k in keys4 if k[0] == 3][0]
s5 = [k for k in keys4 if k[0] == 5][0]
check(4, "streak 3 + 5 → reactions fired", s3[1] != "silent" and s5[1] != "silent"
      and s3[2] == "streak3" and s5[2] == "streak5", keys4)
p3 = P._dance_phase("Bobo", 3, "hit", 40, 4)
p5 = P._dance_phase("Bobo", 5, "hit", 40, 6)
check(4, "escalating tier in the prompt (GROOVE→FLOW)",
      "GROOVE" in p3 and "FLOW STATE" in p5, "streak3→GROOVE, streak5→FLOW STATE")

# ══ TEST 5 — router: premade vs live, live>1s → bank fallback ════════
g5 = P.GameVoiceGate(all_time_best=10)
g5.decide({"event": "music_tick", "sec": 30}, 100.0)
d_first = g5.decide({"event": "hit", "streak": 1, "action": "hand-right"}, 103.0)
check(5, "first-ever hit → LIVE path", d_first["action"] == "live"
      and d_first["key"] == "first_ever", d_first)
routine = []
for i in range(6):
    d = g5.decide({"event": "hit", "streak": 2, "action": "hand-right"}, 107.0 + i * 4)
    routine.append((d["action"], d["key"]))
pm_keys = {k for a, k in routine if a == "premade"}
check(5, "routine hits → premade bank (micro/named), silence mixed in",
      pm_keys.issubset({"micro", "hit_named"}) and any(a == "silent" for a, k in routine)
      and len(pm_keys) >= 1, routine)
d_mic = g5.decide({"event": "mic_text", "text": "what is your name?"}, 140.0)
check(5, "kid micText question → LIVE path", d_mic["action"] == "live"
      and d_mic["key"] == "micText_q", d_mic)


async def _t5():
    async def slow():
        await asyncio.sleep(1.5)
        return "ooh you FLEW!"
    async def fast():
        await asyncio.sleep(0.1)
        return "ohh you did it!!"
    async def unsafe():
        return "Great job, you are amazing!"
    l1 = await P.speak_live_or_bank(slow, "first_ever", timeout=1.0)
    l2 = await P.speak_live_or_bank(fast, "first_ever", timeout=1.0)
    l3 = await P.speak_live_or_bank(unsafe, "first_ever", timeout=1.0)
    return l1, l2, l3

l_slow, l_fast, l_unsafe = asyncio.get_event_loop().run_until_complete(_t5()) \
    if sys.version_info < (3, 10) else asyncio.run(_t5())
check(5, "live >1000ms → bank fallback fired", l_slow[1] == "bank_fallback" and l_slow[0],
      f"slow→{l_slow} fast→{l_fast}")
check(5, "live line rails: banned praise → bank covers", l_unsafe[1] == "bank_fallback"
      and "great job" not in l_unsafe[0].lower(), l_unsafe)

# ══ TEST 6 — detection dies → dance-along, zero 'I saw' ══════════════
g6 = P.GameVoiceGate()
g6.decide({"event": "music_tick", "sec": 30}, 100.0)
g6.decide({"event": "detection", "ok": False}, 101.0)
d_hit = g6.decide({"event": "hit", "streak": 1}, 104.0)
d_mm = g6.decide({"event": "music_moment", "label": "drop"}, 108.0)
saw = [x for x in P.GAME_BANKS["dancealong"] + P.GAME_BANKS["music_moment"]
       + P.GAME_BANKS["section"] if P._P3_SAW.search(x)]
check(6, "detection dead → body events silent, song-reacts only",
      d_hit["action"] == "silent" and "no_detection" in d_hit["reason"]
      and d_mm["action"] == "premade" and d_mm["key"] == "dancealong",
      f"hit→{d_hit['action']}({d_hit['reason']}) moment→{d_mm['key']}")
check(6, "'I saw' BANNED with no data (banks + sanitizer)",
      not saw and P.sanitize_game_line("I saw that!", detection_ok=False) == "",
      f"bank offenders={saw} sanitizer→''")

# ══ TEST 7 — voice dies → game completes on lights, no errors ════════
g7 = P.GameVoiceGate(all_time_best=10)
g7.decide({"event": "music_tick", "sec": 30}, 100.0)
g7.decide({"event": "hit", "streak": 1}, 103.0)      # burn first-ever
g7.voice_ok = False                                   # ← kill voice (as _say_game_line would)
errs = []
quiet = []
try:
    for i, ev in enumerate([{"event": "hit", "streak": 2}, {"event": "miss"},
                            {"event": "music_moment"}, {"event": "hit", "streak": 2},
                            {"event": "singing"}, {"event": "away"}]):
        d = g7.decide(ev, 110.0 + i * 4)
        if ev["event"] not in ("away",):              # away/milestones may retry (rejoin path)
            quiet.append((ev["event"], d["action"]))
except Exception as e:
    errs.append(repr(e))
routine_all_silent = all(a == "silent" for n, a in quiet)
d_mile = g7.decide({"event": "hit", "streak": 3}, 140.0)  # milestone retries → natural rejoin
g7.voice_ok = True                                        # say succeeded
d_after = g7.decide({"event": "music_moment"}, 145.0)
no_gap_comment = "rejoin" not in json.dumps(P.GAME_BANKS)  # no 'I'm back!' bank exists
check(7, "voice dead → zero errors, routine goes silent, lights own the game",
      not errs and routine_all_silent, f"errs={errs} decisions={quiet}")
check(7, "voice back → rejoins via next beat, no comment about the gap",
      d_mile["action"] == "premade" and d_after["action"] == "premade" and no_gap_comment,
      f"milestone_retry={d_mile['action']}/{d_mile['key']} after={d_after['action']}/{d_after['key']}")

# ══ TEST 8 — kid speech: question → 1 line; story → deferred → ending ═
check(8, "classifier: question vs story",
      P.mic_text_kind("what's your name?") == "question"
      and P.mic_text_kind("can you see me") == "question"
      and P.mic_text_kind("I have a cat named Mango") == "story", "3/3")
g8 = P.GameVoiceGate()
g8.decide({"event": "music_tick", "sec": 30}, 100.0)
d_q = g8.decide({"event": "mic_text", "text": "why is the light green?"}, 103.0)
d_s = g8.decide({"event": "mic_text", "text": "I have a cat named Mango"}, 110.0)
deferred = g8.deferred_topics
ending = P._goodbye_phase("Bobo", 8, 5, None, 1, deferred_topic=deferred[0] if deferred else None)
check(8, "question → ONE quick live line; story → tiny ack + deferred",
      d_q["action"] == "live" and d_q["key"] == "micText_q"
      and d_s["key"] in ("micText_ack", None) and deferred
      and deferred[0] == "I have a cat named Mango",
      f"q={d_q['action']}/{d_q['key']} story={d_s['action']}/{d_s['key']} deferred={deferred}")
check(8, "story resurfaces in the ENDING prompt (continuity gold)",
      "cat named Mango" in ending and "CONTINUITY GOLD" in ending,
      "goodbye prompt contains the deferred story + continuity instruction")
d_parent = g8.decide({"event": "mic_text", "text": "dinner is ready honey", "speaker": "adult"}, 120.0)
check(8, "parent voice mid-song → ignored", d_parent["action"] == "silent"
      and d_parent["reason"] == "parent_voice_ignored", d_parent["reason"])

# ══ output ═══════════════════════════════════════════════════════════
passed = sum(1 for c in CHECKS if c["pass"])
print(json.dumps({"checks": CHECKS, "passed": passed, "total": len(CHECKS),
                  "router_log_sample": g1.log[:6]}, ensure_ascii=False))
