# Nova voice fix — "ooo ooo" / "Parkinson" filler + latency

## TL;DR
The filler bug is **already fixed in code and pushed to `main`.** If you still heard the
stuttering "ooo ooo" in a session, the **deployed Render worker is stale or has
`NOVA_FILLERS=1` set in its dashboard.** This is a *deploy/env* fix, not a code change.

## What was wrong, and where it's fixed (all already on `main`)
| Symptom | Root cause | Fix (commit) |
|---|---|---|
| Double "ooo ooo" voice | Browser delivered each kid utterance **twice** (published mic → STT, AND a `user-said` data packet). Both hooks fired a filler. | `20bc064` — atomic `FillerPlayer.claim()` stamps `last_fire` **synchronously** before `create_task(fire)`, so the twin hook sees the gap and skips. `agent.py:196–213`. |
| Trembly "Parkinson" sound | The `ahh.wav` clip is a long open vowel; when the real reply interrupts it mid-vowel it sounds strangled. | `c5619bd` — `ahh` dropped. `FILLER_NAMES = ["mm","mmhm","ooh","ohh","hmm"]` (closed/short sounds that cut cleanly). `agent.py:126`. |
| Filler lands *after* the reply | Every audio clip goes through Runway lipsync (~500ms tax), so a "thinking sound" can't reliably land *before* the reply. | `6e544e0` — fillers **OFF by default**: `os.getenv("NOVA_FILLERS","0")=="1"`. `agent.py:166`. |
| First reply ~3s | Cold OpenAI TLS/connection pool. | `ba79374` — `_warm_llm()` pre-opens the connection at startup → ~1.4s warm. |

## Do this on Render (the actual fix)
1. **Env var:** open the worker service → **Environment**. If `NOVA_FILLERS` exists and
   is `1`, set it to `0` **or delete it** (default is already off). Save.
2. **Redeploy latest `main`:** Manual Deploy → **Deploy latest commit**. Confirm the
   deployed commit is `40d220b` or newer (`git log -1` locally = `40d220b`).
3. **Verify in logs** after a test session — you should NOT see `[filler] CLAIM ...`
   lines. If you do, `NOVA_FILLERS` is still `1` somewhere.

## Latency (for reference — already tuned, no action needed)
- Warm path: STT-final → first audio ≈ **855ms** (700–1100ms).
- Fixed floor: Runway lipsync ≈ **500ms** per audio frame — the bottleneck. "Magic"
  (<500ms) isn't reachable while the head/face avatar is in the pipeline.
- Already in place: Deepgram endpointing 250ms, VAD turn-detection, Flash v2.5 streaming
  TTS, dance-phase replies capped ~80 tokens, LLM pre-warm.

## Frontend note (dance-project)
The old `forceNovaMidDanceVoice()` that hit `/tts-for-mic` (404 on novapython) is
**already disabled** — `smartReact()` routes mid-dance reactions through the worker
brain over the LiveKit data channel (`void shouldForce`). Confirmed in `nova-joined.html`
and `v113-live.html`. The dead function is harmless and unused.
