# Nova Dance — AI Kids' Dance App

> ═══════════════════════════════════════════════════════════════
> ## ⭐ LIVE STATE — updated 2026-06-19 (this section supersedes any older detail below)
> ═══════════════════════════════════════════════════════════════
>
> **USER:** dsbeauty01-web — solo non-programmer, Ko Samui Thailand. Short fragmented English.
> Wants direct fixes. NO yes-man, NO over-explaining, NO diagnostic loops. Make small decisions
> yourself, don't ask mid-sprint. Report when DONE, not when planning. Self-test before sending URL.
>
> **WHAT:** 5-min daily ritual for kids 6-10 — they "play with Nova", a warm magical big-sister AI.
> Voice + camera + simple move game. Tech works, magic doesn't yet. ~8 months in, ZERO real kid tests.
> Target feeling: kid says "can I play with Nova tomorrow?"
>
> ### Deployed stack (working — don't touch the locked bits)
> - **Worker:** `agent.py` on Render service "worker" (paid standard tier)
> - **Frontend (live):** `dsbeauty01-web.github.io/dance-project/v113-live.html`  → local: `C:\Users\ADMIN\projects\dance-project\v113-live.html`
> - **Server:** `novapython.onrender.com` (free tier — cold start 30-50s)
> - **VOICE (LOCKED):** ElevenLabs Flash v2.5, Loora `P6xfJudBtfcB1BM5ZWR7` — stability 0.65, similarity 0.90, style 0.30, speed 0.92, speaker_boost True
> - **STT (LOCKED, just deployed):** Deepgram Nova-3 server-side (browser SpeechRecognition disabled). model nova-3, lang en, smart_format True, endpointing 400ms. Tool calling still works.
> - **LLM:** OpenAI gpt-4o-mini, temp 0.85, 5-layer system prompt
> - **AVATAR (LOCKED):** Runway custom Pixar Nova `e976bbb2-de60-4da6-845e-4b754050e55b`. HEAD + FACE ONLY (no body/hands). ~500ms lipsync tax.
> - **VISION (LOCKED):** Gemini 2.5 Flash Lite via `/v2/vision-observe` (SDK now `google.genai`, was `google.generativeai`)
> - **FILLERS:** 8 cached clips in `/audio/fillers/` — currently DISABLED on worker (caused 5-10s blocks)
> - **Latency (warm):** STT-final → first audio ~855ms (700-1100ms). Bottleneck = Runway lipsync ~500ms fixed. Magic = <500ms.
>
> ### DO NOT TOUCH: Loora voice/settings · Deepgram STT · `/v2/vision-observe` · Runway UUID e976bbb2 · 5-layer prompt structure · worker-v222 (suspended, OpenAI is the brain)
>
> ### NOVA PERSONA (locked)
> Cool magical big-sister, 11-12 feel. 110% more excited than the kid. Smile-in-voice every reply.
> Specific not generic ("your right hand was HIGH!" not "good job"). Mirror kid's name/words with delight.
> Replies 1-2 sentences MAX; in-game reactions 4-7 words.
> - **BANNED:** "great job", "amazing", "awesome", "perfect", "ok ok", "hahah", "yeah yeah", "right right", "mm mm", generic positive filler
> - **ALLOWED (max 1/reply):** "Yo", "YESSS", "WHOA", "BOOM", "okay okay!", "ohh", "mhm", "wait...", "huh", "hmm", "ohh yes", "oh!"
>
> ### THE GAME — MOVE PLAY (designed, NOT YET BUILT) ← priority #1
> Kids 6-10. Nova watches via camera, reacts to moves (she can't move her body, so the kid is the star).
> 5-min MAX, flexible 2-5min. Flow:
> 1. Arrival 15s: "Yo! Hi! I'm Nova! What should I call you?"
> 2. Invite 15s: "[name]! ready to play?"
> 3. Moves loop (until done or 5min): Nova picks a prompt → kid does it → vision fires → Nova reacts specific → "wanna do another?" / auto-continue if engaged
> 4. Exit anytime: reference one thing kid did → "come back whenever — I'll be here"
> 5. HARD CAP 5min auto-end
> - **Move library (easy→hard):** wave at me / raise one hand HIGH / clap 3 times / spin once / wave like a tree / make yourself BIG / freeze like a statue 3s / show me your COOLEST move
> - **Vision-reaction per move:** Nova says prompt → wait 2s → vision fires → LLM gets "Kid asked to [PROMPT]. Vision sees [OBS]. React 1 short specific warm sentence." → Loora speaks
> - **Safety:** silent 15s → playful nudge "hey [name]? still there?"; 3 nudges → graceful goodbye; 5min hard cap
>
> ### KNOWN BUGS
> 1. `forceNovaMidDanceVoice` 404 — old endpoint kills vision speak → route through normal `speak()`, delete the 404 call
> 2. Cold start 30-50s (free Render) → paid tier OR keep-alive cron ping
> 3. Idle nudge ghost loop fires after session ends → cancel idle task in session close callback
> 4. Audio + video tracks arrive 1s+ apart → wait for both subscribed before showing Nova
> 5. Boot warnings in `agent.py`: allow_interruptions deprecated (→TurnHandlingOptions); RoomInput/OutputOptions deprecated (→RoomOptions); resume_false_interruption unsupported on audio path
>
> ### NEXT TASKS (priority order)
> 1. BUILD the move-play game (spec above) — prompts library, vision-reaction loop, 5min cap, nudge logic
> 2. CLEAN bugs (forceNovaMidDanceVoice 404 first)
> 3. POLISH intro reveal — readiness gate (don't show Nova until ALL ready), reveal beat (fade + hold 1.2s + warm expression + speak), slow greeting with pauses
> 4. SELF-TEST in incognito before reporting
>
> ═══════════════════════════════════════════════════════════════

## Project Overview
Nova Dance is an AI-powered dance application designed for kids. It combines real-time dance detection, avatar animation, and interactive music-based gameplay.

## Current Architecture

### Frontend (dance-project repo)
- **v208+ HTML** — Latest frontend versions (v209 planned)
- Location: `c:/Users/ADMIN/projects/dance-project/`
- Single-file HTML architecture, versioned for rapid iteration
- Key UI states:
  - ARRIVAL — soft pulsing dot, kid taps to start
  - RECOGNITION — Nova greeting, kid intro
  - DANCE — main gameplay (webcam, music, detection)
  - GOODBYE — farewell, streak summary
- Components:
  - Avatar face (Rive animations from nova.riv)
  - Webcam video (pose detection, dance frame check)
  - Music player with Runway face lipsync
  - Score/streak display
  - Parent/admin inspection pages (separate HTML files)

### Backend (novapython worker)
- **v207 Brain** — Loora-grade AI intelligence backend (OpenAI gpt-4o-mini)
- Location: `c:/Users/ADMIN/projects/novapython/`
- **5-Layer Prompt Architecture:**
  - L1 IDENTITY (locked, ~500t) — Nova's persona: 20yo dance friend, warm, alive
  - L2 KID PROFILE (per-kid) — Name, sessions, streaks, shared facts from memory
  - L3 KID KNOWLEDGE (on-demand) — Colors, animals, foods, context from knowledge.py
  - L4 SESSION STATE (live) — Phase, music sec, recent events, message history
  - L5 PHASE PERSONA (locked) — Behavior: recognition / dance / goodbye

- **Reaction Tiering (Router):**
  - Tier 1: phrase_bank (~50ms, 80% of reactions)
  - Tier 2: llm_micro (~500ms, milestone streaks 3/5/10, first hits)
  - Tier 3: llm_rich (~700ms, kid speech, goodbye, vision)

- **Core modules:**
  - `agent.py` — LiveKit agent worker, orchestrates pipeline
  - `personality.py` — 5-layer prompt builder, phrase bank, reaction tier router
  - `memory.py` — Postgres (Render) or RAM fallback, per-kid facts/history
  - `knowledge.py` — Knowledge base (colors/animals/foods/etc.)
  - `vision.py` — Gemini vision for dance detection/feedback
  - `server.py` — FastAPI, issues LiveKit tokens, hosts memory/vision endpoints
  - `requirements.txt` — Dependencies

- **Pipeline:**
  Kid voice (Web Speech API) → LiveKit → worker → gpt-4o-mini → ElevenLabs Freya TTS → Runway face lipsync → kid

## Integration Points
- Frontend sends video/motion data → Backend processes via vision.py
- Backend returns dance feedback, music cues, character responses
- Memory system tracks user progress and preferences
- Personality system generates context-aware interactions

## Development Status
- **Frontend:** Versioned iteratively (v100+ range, latest v208, v209 ready to build)
  - UI/UX: Mature (pulsing arrival, face display, stage transitions)
  - Rive avatar system: Integrated
  - Runway face lipsync: Active
  - Webcam/pose detection: Implemented
- **Backend:** Stable Loora-grade implementation (v207, OpenAI gpt-4o-mini)
  - 5-layer prompts: Locked and tuned
  - Reaction tiering: Deployed (phrase bank 80% of reactions)
  - Memory system: Working (Postgres on Render or RAM fallback)
  - Knowledge base: Ready (colors/animals/foods/etc.)
  - Vision pipeline: Live (Gemini observation → feedback)

## Active Improvements (v207→v209)
- Dance detection accuracy refinement
- Personality depth (mirror-and-echo, gentle corrections)
- Phase persona transitions (recognition/dance/goodbye)
- Memory quality (shared_facts, best_moments tracking)

## How to Work on This
1. **Quick frontend tweaks:** Edit v208.html, version to v209.html
2. **Backend logic:** Modify personality.py (phrase_bank, reaction_tier), agent.py (pipeline)
3. **Memory/knowledge:** Update memory.py or knowledge.py per-kid facts
4. **Debugging:** Check [HEAR]/[TYPE]/[BRAIN]/[SPEAK]/[PACKET] log tags in agent.py
5. **Deployment:** Render auto-deploys from git; LiveKit tokens issued from server.py
