"""Machine-check of THE ENDING engine (NOVA-ENDING.md self-test 1-7, data layer)."""
import personality as P
f = []

# 1) length scaling: wave is quick (3 lines path), others full (4)
assert P.GOODBYE_SCORES["wave"]["quick"] is True
assert all(not P.GOODBYE_SCORES[s]["quick"] for s in ("hello", "joined", "freeze"))

# 2) callback = REAL events only
assert "FREEZE" in P.pick_goodbye_callback("hello", {"freeze"}, 5)
assert P.pick_goodbye_callback("hello", {"left"}, 5) == P.GOODBYE_BRAVERY  # no matching action -> bravery
assert P.pick_goodbye_callback("hello", set(), 0) is None                  # zero hits -> caller uses techblame
assert "WHOLE arm" in P.pick_goodbye_callback("wave", {"wristwave"}, 3)
assert "FLYING" in P.pick_goodbye_callback("joined", {"combo", "shrug"}, 8)  # first matching wins

# 3) deposit priorities + intro opener
l, k, i = P.pick_deposit("hello", None, False, None)
assert k == "finish:hello" and "finish" in l and "finish" in i.lower() or "YES" in i
l, k, i = P.pick_deposit("hello", "my cat Mango", True, None)
assert k == "topic" and "tomorrow you tell me" in l
l, k, i = P.pick_deposit("hello", None, True, None)
assert k.startswith("tease:") and "next time" in l

# 4) never the same deposit twice in a row
l1, k1, _ = P.pick_deposit("wave", None, True, None)
l2, k2, _ = P.pick_deposit("wave", None, True, k1)
assert k1 != k2, f"deposit repeated: {k1}"
# and unfinished doesn't repeat either
_, kf1, _ = P.pick_deposit("wave", None, False, "finish:wave")
assert kf1 != "finish:wave"

# 5) every game has a tease target that exists
for s, (line, nxt) in P.NEXT_GAME_TEASE.items():
    assert nxt in P.SONG_DUR, f"tease target {nxt} unknown"

print("ENDING SELF-TEST: ALL PASS")
