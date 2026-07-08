# MILESTONE 1 — "text works, gemini works" (2026-07-08, user-declared)

The first fully-working commercial intro: voice AND typed chat both get her
spoken reply, whole intro flow runs end to end. Restore this exact state with
the tags + env below.

## What works here (user-verified live + probe-verified)
- COMMERCIAL-INTRO architecture: producer=whisperer (no generate_reply in
  conversation), turn-engine beats, 2-cue light challenge chain, air rule,
  STT-echo bubble, state badges, cue pulse ring, whispered game transition.
- TYPED CHAT → her voice: "im bobo" → "Hi Bobo — your magic friend! ...ready
  to make a move?" in 1.13s ([LAT] inside the 1.5s spec target).
- Voice conversation via **Gemini Live** (Hume credits were out — 3rd time).
- Mic-denied session completes the whole intro by typing only.

## Pinned code
- novapython: tag `milestone-1` (commit 1bb8613)
- dance-project: tag `milestone-1` (commit 5d4c64a, page boot line
  "v116 COMMERCIAL-INTRO")

## Worker env (Render srv-d8c20euq1p3s73ft6aog) — the deciding flags
```
USE_GEMINI=1                 # ← the active voice: Gemini Live (wins over EVI)
USE_EVI=1                    # Hume wiring intact underneath
NOVA_FORCE_ELEVENLABS=0
NOVA_V2V=1
HUME_CONFIG_ID=bddc965a-0b47-44f3-97b8-37ce89526d65
# defaults in code: NOVA_GEMINI_MODEL=gemini-3.1-flash-live-preview,
#                   NOVA_GEMINI_VOICE=Leda, NOVA_CLIPS=1
```

## Known accepted tradeoffs at this milestone
- Clips are Kora's voice, live talk is Gemini "Leda" → voice mismatch.
- Whispers on Gemini are buffered (ride the next turn), not instant.
- COMMERCIAL-INTRO-GOLD not tagged yet (self-test 9 "one person" blocked by
  the voice mismatch).

## To return to Hume/Kora later
Fix billing at platform.hume.ai/billing → delete USE_GEMINI (or set 0) on the
worker → restart. Nothing else changes.

## Other state
- worker-v222: SUSPENDED (was crash-looping).
- Deepgram: deleted everywhere (code + keys) earlier the same day.
