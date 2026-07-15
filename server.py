"""
Nova v200 — HTTP API server (complete: Days 1-5)

The lightweight FastAPI service the browser talks to.

Responsibilities:
1. Issue LiveKit room tokens so browser can join (/v2/create-session)
2. Vision endpoint — accept webcam frame, send to Gemini, return observation (/v2/vision-observe)
3. Memory endpoint — get/set per-kid memory (/v2/memory)

The actual conversation happens between the browser and the LiveKit Agent
(see agent.py). The agent worker handles STT/LLM/TTS and is always running.
"""

import os
import time
import json
import hashlib
import secrets
import logging
from datetime import timedelta
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from livekit import api

import memory
import vision

load_dotenv()

logger = logging.getLogger("nova-v200-api")
logging.basicConfig(level=logging.INFO)

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Nova v200 API starting up")
    logger.info(f"  LIVEKIT_URL: {bool(LIVEKIT_URL)}")
    logger.info(f"  ANTHROPIC: {bool(os.getenv('ANTHROPIC_API_KEY'))}")
    logger.info(f"  RUNWAY: {bool(os.getenv('RUNWAYML_API_SECRET'))}")
    logger.info(f"  ELEVENLABS: {bool(os.getenv('ELEVENLABS_API_KEY'))}")
    logger.info(f"  GEMINI: {bool(os.getenv('GEMINI_API_KEY'))}")
    yield
    logger.info("Nova v200 API shutting down")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "ok": True,
        "service": "nova-v200-api",
        "version": "v200-complete",
        "tagline": "one brain · livekit agents · runway face",
    }


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/v2/diag")
async def diag():
    """Diagnostic info for the test page — keys + recent activity."""
    # A flaky memory backend must never take down the diagnostics panel.
    try:
        total_kids = len(memory.store.all_kids())
    except Exception as e:
        logger.warning(f"[diag] all_kids() failed: {e}")
        total_kids = f"err: {e}"
    return {
        "ok": True,
        "service": "nova-v200-api",
        "keys": {
            "LIVEKIT_URL": bool(LIVEKIT_URL),
            "LIVEKIT_API_KEY": bool(LIVEKIT_API_KEY),
            "LIVEKIT_API_SECRET": bool(LIVEKIT_API_SECRET),
            "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
            "ELEVENLABS_API_KEY": bool(os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_API_KEY")),
            "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY")),
            "GOOGLE_API_KEY": bool(os.getenv("GOOGLE_API_KEY")),
            "HUME_API_KEY": bool(os.getenv("HUME_API_KEY")),
            "LEMONSLICE_API_KEY": bool(os.getenv("LEMONSLICE_API_KEY") or os.getenv("LEMONSILCE_API_KEY") or os.getenv("lemonsilce") or os.getenv("lemonslice")),
            "RUNWAYML_API_SECRET": bool(os.getenv("RUNWAYML_API_SECRET")),
        },
        # VOICE-SILENCE DEBUG (2026-07-09): which mouth is the worker actually
        # wired to? These flags decide it in agent.py — silent voice with
        # working clips means the selected backend fails per-turn.
        "voice_flags": {
            "USE_GEMINI": os.getenv("USE_GEMINI", ""),
            "USE_EVI": os.getenv("USE_EVI", ""),
            "NOVA_V2V": os.getenv("NOVA_V2V", ""),
            "NOVA_FORCE_ELEVENLABS": os.getenv("NOVA_FORCE_ELEVENLABS", ""),
            "NOVA_GEMINI_MODEL": os.getenv("NOVA_GEMINI_MODEL", "(default)"),
            # fingerprints (first 10 hex of sha256) — compare keys across
            # machines without ever exposing them
            "gemini_key_fp": hashlib.sha256(os.getenv("GEMINI_API_KEY", "").encode()).hexdigest()[:10],
            "google_key_fp": hashlib.sha256(os.getenv("GOOGLE_API_KEY", "").encode()).hexdigest()[:10],
        },
        "total_kids": total_kids,
    }


# ────────────────────────────────────────────────────────────────────────
# /v2/create-session — issue LiveKit room token to browser
# ────────────────────────────────────────────────────────────────────────
class CreateSessionReq(BaseModel):
    kidId: Optional[str] = None
    kidName: Optional[str] = None
    agent: Optional[str] = "nova"   # v222 sends "nova222" to reach the Gemini worker
    voiceOnly: Optional[bool] = False   # ?voiceonly pages: Nova voice, NO Runway avatar
    directGame: Optional[str] = None    # ?game= pages: skip intro (name/challenge) — game voice only
    # TASK 1 (fable5): the page PREFETCHES the session at load with dispatch=False
    # (room+token only). Nova is summoned later via /v2/dispatch at the orb tap —
    # dispatching her into an empty room at page load left her tracks unpublished.
    dispatch: Optional[bool] = True
    # HEBREW (?lang=he): carried in room+dispatch metadata → agent builds the
    # Hebrew persona block and turns clips off. Default "en" = behavior unchanged.
    lang: Optional[str] = None


@app.post("/v2/create-session")
async def create_session(req: CreateSessionReq):
    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        raise HTTPException(500, "LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET not set")

    room_name = f"nova-{int(time.time())}-{secrets.token_hex(4)}"
    identity = f"kid-{secrets.token_hex(4)}"

    # Stable kid_id (browser sends, or we generate)
    kid_id = req.kidId or f"anon-{secrets.token_hex(6)}"

    # Embed kid_id (+ mode flags) in room metadata so the agent picks it up
    room_metadata = json.dumps({"kidId": kid_id, "voiceOnly": bool(req.voiceOnly),
                                "directGame": req.directGame or None,
                                "lang": (req.lang or "en")})

    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(req.kidName or "kid")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,  # browser sends game events via data
            )
        )
        .with_ttl(timedelta(minutes=10))
        .to_jwt()
    )

    # Pre-create room AND explicitly dispatch the "nova" agent into it.
    # The explicit dispatch is the reliable way — the token-embedded version
    # was being silently ignored.
    try:
        livekit_api = api.LiveKitAPI(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        await livekit_api.room.create_room(
            api.CreateRoomRequest(name=room_name, metadata=room_metadata)
        )
        if req.dispatch:
            # Summon the requested worker into this room (default "nova"; v222 sends
            # "nova222" for the Gemini-brain worker). This is what makes it JOIN.
            agent_name = req.agent or "nova"
            dispatch = await livekit_api.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name=agent_name,
                    room=room_name,
                    metadata=room_metadata,
                )
            )
            logger.info(f"[dispatch] {agent_name} → room={room_name} id={dispatch.id}")
        else:
            logger.info(f"[dispatch] DEFERRED (prefetch) room={room_name} — /v2/dispatch will summon")
        await livekit_api.aclose()
    except Exception as e:
        logger.error(f"[dispatch] FAILED to summon nova: {e}")

    logger.info(f"[create-session] room={room_name} kid_id={kid_id}")

    return {
        "ok": True,
        "roomName": room_name,
        "token": token,
        "serverUrl": LIVEKIT_URL,
        "identity": identity,
        "kidId": kid_id,
    }


# ────────────────────────────────────────────────────────────────────────
# /v2/dispatch — summon Nova into an already-created room (prefetch flow).
# The page calls this AT THE ORB TAP, so the agent always joins a room the
# kid is entering — never sits alone in an empty room from page load.
# ────────────────────────────────────────────────────────────────────────
class DispatchReq(BaseModel):
    roomName: str
    kidId: Optional[str] = None
    agent: Optional[str] = "nova"
    lang: Optional[str] = None   # HEBREW: tap-dispatch carries lang too (job metadata wins pre-connect)


@app.post("/v2/dispatch")
async def dispatch_agent(req: DispatchReq):
    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        raise HTTPException(500, "LiveKit env not set")
    room_metadata = json.dumps({"kidId": req.kidId or "", "lang": (req.lang or "en")})
    try:
        livekit_api = api.LiveKitAPI(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        dispatch = await livekit_api.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=req.agent or "nova",
                room=req.roomName,
                metadata=room_metadata,
            )
        )
        await livekit_api.aclose()
        logger.info(f"[dispatch] (tap) {req.agent} → room={req.roomName} id={dispatch.id}")
        return {"ok": True, "dispatchId": dispatch.id}
    except Exception as e:
        logger.error(f"[dispatch] (tap) FAILED: {e}")
        raise HTTPException(500, f"dispatch failed: {e}")


# ────────────────────────────────────────────────────────────────────────
# /v2/vision-observe — Gemini Flash describes a webcam frame
# ────────────────────────────────────────────────────────────────────────
class VisionReq(BaseModel):
    frameDataUrl: str   # data:image/jpeg;base64,...
    kidId: Optional[str] = None


@app.post("/v2/vision-observe")
async def vision_observe(req: VisionReq):
    if not req.frameDataUrl or not req.frameDataUrl.startswith("data:image/"):
        raise HTTPException(400, "frameDataUrl missing or invalid")

    observation = await vision.observe_from_data_url(req.frameDataUrl)
    if not observation:
        return {"ok": False, "text": None}

    return {"ok": True, "text": observation}


# /v2/age-estimate — indirect age read from a webcam frame (Gemini → a/b/c/d → tier)
_AGE_TIER = {"a": "LITTLE", "b": "KID", "c": "TEEN", "d": "ADULT"}

@app.post("/v2/age-estimate")
async def age_estimate(req: VisionReq):
    if not req.frameDataUrl or not req.frameDataUrl.startswith("data:image/"):
        raise HTTPException(400, "frameDataUrl missing or invalid")
    letter = await vision.estimate_age(req.frameDataUrl)
    return {"ok": True, "letter": letter, "tier": _AGE_TIER.get(letter, "KID")}


# ────────────────────────────────────────────────────────────────────────
# /v2/memory/{kid_id} — get current memory for a kid
# ────────────────────────────────────────────────────────────────────────
@app.get("/v2/memory/{kid_id}")
async def get_memory(kid_id: str):
    return {"ok": True, "memory": memory.store.to_dict(kid_id)}


class MemoryUpdateReq(BaseModel):
    kidId: str
    name: Optional[str] = None
    age: Optional[int] = None
    favoriteMove: Optional[str] = None


@app.post("/v2/memory/update")
async def update_memory(req: MemoryUpdateReq):
    fields = {k: v for k, v in req.dict().items() if k != "kidId" and v is not None}
    # Convert camelCase to snake_case
    if "favoriteMove" in fields:
        fields["favorite_move"] = fields.pop("favoriteMove")
    mem = memory.store.update(req.kidId, **fields)
    return {"ok": True, "memory": memory.store.to_dict(req.kidId)}


# ────────────────────────────────────────────────────────────────────────
# /v2/stats — for debugging
# ────────────────────────────────────────────────────────────────────────
@app.get("/v2/stats")
async def stats():
    kids = memory.store.all_kids()
    return {
        "ok": True,
        "total_kids": len(kids),
        "kids": [
            {
                "kidId": k.kid_id,
                "name": k.name,
                "totalSessions": k.total_sessions,
                "maxStreak": k.max_streak,
            }
            for k in kids
        ],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
