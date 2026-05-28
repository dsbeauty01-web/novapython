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
    logger.info(f"  DEEPGRAM: {bool(os.getenv('DEEPGRAM_API_KEY'))}")
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


# ────────────────────────────────────────────────────────────────────────
# /v2/create-session — issue LiveKit room token to browser
# ────────────────────────────────────────────────────────────────────────
class CreateSessionReq(BaseModel):
    kidId: Optional[str] = None
    kidName: Optional[str] = None


@app.post("/v2/create-session")
async def create_session(req: CreateSessionReq):
    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        raise HTTPException(500, "LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET not set")

    room_name = f"nova-{int(time.time())}-{secrets.token_hex(4)}"
    identity = f"kid-{secrets.token_hex(4)}"

    # Stable kid_id (browser sends, or we generate)
    kid_id = req.kidId or f"anon-{secrets.token_hex(6)}"

    # Embed kid_id in room metadata so the agent picks it up
    room_metadata = json.dumps({"kidId": kid_id})

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

    # Pre-create room with metadata (agent worker dispatches on this)
    try:
        livekit_api = api.LiveKitAPI(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        await livekit_api.room.create_room(
            api.CreateRoomRequest(name=room_name, metadata=room_metadata)
        )
        await livekit_api.aclose()
    except Exception as e:
        logger.warning(f"pre-create room failed (will use default): {e}")

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
