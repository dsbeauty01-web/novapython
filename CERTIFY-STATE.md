# NOVA-CERTIFY — working state (2026-08-08, session scratch — do not ship)

## Mission
Voice-only certification of nova-commercial.html `?voiceonly` path + novapython worker.
Full mission text lives in the founder's message (Stages 1-4, P1-P8/G1-G6/T1-T4/E1-E5,
fix per BRAIN-6-FIXES.md, retest x3, one report then HOLD).

## Hard findings so far
1. `BRAIN-6-FIXES.md` + `FINISH-IT-ALL.md` DO NOT EXIST — not in dance-project, not in
   novapython, not in git history of either, not in Downloads. Past sessions (hosty)
   also searched and failed. → Substitute source of truth: the PROVEN gate
   implementations in dance-project/pod/rt_lk.py ([TURN-GATE][CONSENT][GARBLE][LIGHT]
   [STATUE][TRUTH-GATE] all live there). Flag as QUESTION in the report.
2. NONE of the 5 required markers exist in novapython/agent.py (grep "[TURN-GATE" etc.
   = zero hits). GARBLE exists only as prompt text (line ~2516), not a code gate.
3. Voice-only brain = novapython worker (Render). agent.py (4987 lines, LiveKit worker,
   OpenAI gpt-4o-mini 5-layer + realtime). Modern wiring = DIRECTOR-GOLD:
   nova_director.py (Director/Scenes/MagicLight/PERSONA_TEXT — read, understood).
   run_friend_intro in agent.py is LEGACY producer ("dead code under DIRECTOR-GOLD").
4. Python 3.12.10 installed on laptop at
   C:\Users\dsbea\AppData\Local\Programs\Python\Python312\python.exe (winget, was missing).
5. Existing sim pattern to copy: novapython/phase3_sim.py (stubs, simulated clock,
   pure stdlib, JSON verdict blob) — machine-checks personality.py.
6. Laws: dance-project tools/laws/run-all.sh must stay green (12/12 currently).
7. DONE ALREADY: brevity law reverted to freeze-only in pod/rt_lk.py (commit 7fb2534
   on freeze-v2, pushed). Pods all stopped.

## Where the certification probes map
- Stage 1 intro: Director scenes intro/light + agent.py turn machinery + picker consent.
- Stage 2 games: personality.py GameVoiceGate/router (phase3_sim covers some) + agent.py.
- Stage 4 ending: server.py /pulse (novapython) — PULSE commit exists (b359481).

## HARNESS WORKS (2026-08-08 ~late)
- certify_sim.py drives the LIVE Render worker (worker IS deployed: env-diag commit
  d868db5e0e, V2V=1, USE_GEMINI=1, USE_EVI=1, NOVA_FRIEND default=1, avatar=pod).
- KEY HANDSHAKE: greet fires ONLY on {"kind":"reveal-now"} packet (INTRO-FINAL law).
- Her words arrive as {"kind":"nova-said"} packets; stage decisions as stage-diag.
- P1 = PASS on live worker: ONE greet line ("Hi! I'm Nova, your magical AI dance
  teacher! What's your name?"), then full silence 45s — zero monologue. (Spec wanted
  exactly one re-invite at 20s; she gives zero — quiet side of the bar, note in report.)
- Full Stage 1 run (P1,P2,P3,P4,P6,P7) in flight. P5 (light once-ever) + P8
  (readiness tasks) still to write.

## STAGE 1 FIRST PASS (live worker, before fixes)
- P1 PASS (one greet line, zero monologue, quiet 45s)
- P2 PASS ("Hey Bobo, awesome to meet you!" — name echoed)
- P3 FAIL — REAL BUG: light fires visually (cue-part packet) but she NEVER speaks
  about it: system items don't trigger realtime generation. P4's "pass" = same mute.
- P6 PASS on consent discipline (no self-pick, quiet at the beat) — but she offered
  "arm circles", NOT the three games (the 3-game offer needs the picker-fact nudge).
  NOTE: director's picker fact lists 'Hello Hello!/Up Groove!/Wave!' — the MISSION
  says Freeze/Wave/Up Groove. Product/mission mismatch — report as QUESTION.
- P7 FAIL — REAL BUG: garble "Peso" became her name ("hi peso! love that name").

## FIXES SHIPPED (novapython main -> Render auto-deploy)
- c8092ca GARBLE WALL: _is_garble (ported verbatim from pod rt_lk.py) gates
  user-said, mic_text facts, name capture. Logs [GARBLE] ignored + stage-diag
  "garble-ignored" packet (harness-visible).
- 377e353 MUTE FIX: Note.send/Director.fact grow nudge=False; nudge=True ONLY on
  light-appear, light-win, picker-offer -> session.generate_reply (one beat).
  Never on barge-in/bookkeeping. Monologue gates untouched.
- Retest battery in flight: P3 P4 P7 P7 P7 P5 (env-diag commit in logs verifies
  the deploy landed; expect 377e353 prefix).

## RETEST AFTER FIXES (commit 377e353 live, verified via env-diag)
- P3 PASS: "whoa, bobo, look at that! there's a magic light on your right shoulder—
  let's try a tiny shoulder wiggle!" -> shrug fact -> "bobo, you rocked it—that's an
  isolation!" — full light beat ALIVE in voice-only for the first time.
- P7 PARTIAL: É + 谢谢你 code-gated (silent, 1 reply/run not 3). Bare "Peso" still
  adopted — lexically indistinguishable from a legit one-word name answer ("Bobo").
  QUESTION for founder: confirmation flow ("Peso — did I hear that right?") = new
  architecture, not built. _extract_name fallback path is the adoption point.
- P5 was FAIL (light re-fired on phase bounce) -> FIX SHIPPED 6d08f92: MagicLight
  'ever' lock, logs [LIGHT] already-done. Retest x3 in flight.
- E5 PASS: /pulse round-trip {"ok":true,"id":...,"score":3}.
- FINAL BATTERY in flight (bglzn4sjd): P5 x3 + G(hello/joined/wave) + T2 + T4.

## Report mapping decisions
- T1 (<=3s dead air) — extract from G-* logs: picked->loading line and song_start->go-line gaps.
- T3 (mid-game silence check-in) — the G script's 70-95s fact-free zone.
- E1-E4 — from G/T4 ending lines (fact-built celebration, feedback ask, named goodbye;
  emoji screen = browser-side, N/A worker).
- P8 as specced (readiness task per game) does NOT exist in the DIRECTOR voice-only
  flow (pick -> loading -> song start; no "show me a freeze" beat) — report N/A with
  the pick->go evidence from G runs. Voice-only picker games are hello/joined/wave
  (NOT freeze) — mission/product naming mismatch, QUESTION.

## Next steps (in order)
1. Find where Director is wired in agent.py (grep Director(/enter_scene) + how
   ?voiceonly picks the agent path (server.py create-session, agent param).
2. Write certify_sim.py in novapython: stub livekit/openai/elevenlabs; fake
   AgentSession recording her lines; scripted kid turns + injected facts + garble;
   simulated clock. Model on phase3_sim.py.
3. Run Stage 1 P1-P8 as-is → collect FAILs (expected: gates missing).
4. Port the 6 gates from pod/rt_lk.py into agent.py/director wiring as CODE gates
   with the exact log markers. Named-file git adds only. Laws stay green.
5. Retest x3 per fixed probe. Stages 2-4. Report table + one clean transcript +
   git log --oneline. Then HOLD.

## Branch/commit discipline
- novapython: work on a branch `certify` (create). dance-project: freeze-v2 already
  carries the revert.
- Report file goes to C:\Users\dsbea\Downloads (founder convention).
