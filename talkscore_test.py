"""Machine-check of the per-song TALK SCORES (NOVA-SONG-TALK-SCORES.md self-test)."""
import personality as P

DUR = {"hello": 111, "wave": 28.5, "joined": 90, "freeze": 60}
fails = []

for song, sc in P.TALK_SCORES.items():
    beats = sc["beats"]
    gap = float(sc.get("min_gap", 2.5))
    # 1) sorted + inside the song
    ts = [t for t, _ in beats]
    if ts != sorted(ts): fails.append(f"{song}: beats not sorted")
    if any(t < 0 or t > DUR[song] for t in ts): fails.append(f"{song}: beat outside song duration")
    # 2) no beat LANDS inside a silence window
    for t, ref in beats:
        if P.talk_in_silence(song, t): fails.append(f"{song}: beat {t}s lands inside silence window")
    # 3) spacing respects the song's gap cap (score must not drop its own beats)
    for a, b in zip(ts, ts[1:]):
        if b - a < gap - 0.05: fails.append(f"{song}: beats {a}->{b} closer than gap {gap}s (would drop)")
    # 4) every @pool exists
    for _, ref in beats:
        if ref.startswith("@") and ref[1:] not in P.TALK_POOLS: fails.append(f"{song}: pool {ref} missing")
    # 5) echo policy valid
    e = sc.get("echo", {})
    if e and e["pool"] not in P.TALK_POOLS: fails.append(f"{song}: echo pool missing")
    print(f"{song:8s} beats={len(beats):2d} gap={gap} silence={sc.get('silence')} echo=1/{e.get('every')}")

# 6) spec anchors
w = dict(P.TALK_SCORES["wave"]["beats"])
assert P.talk_in_silence("wave", 20.0), "wave chain 18-22.15 must be silent"
assert P.talk_in_silence("joined", 55.0), "upgroove 50-59.5 must be silent"
assert P.talk_in_silence("hello", 80.0), "hello fast verse must be silent"
assert any(t <= 18.0 and "wave" in str(r).lower() or r == "@chain_open" for t, r in P.TALK_SCORES["wave"]["beats"]), "chain opener"

# 7) anti-repeat: 20 picks never repeat either of the previous two
used = {}
picks = [P.talk_pool_pick("@hit_echo", used) for _ in range(20)]
for i in range(2, len(picks)):
    if picks[i] in (picks[i-1], picks[i-2]): fails.append(f"anti-repeat broke at pick {i}: {picks[i-2:i+1]}")

print()
if fails:
    print("FAILURES:"); [print(" -", f) for f in fails]; raise SystemExit(1)
print("TALK-SCORE SELF-TEST: ALL PASS")
