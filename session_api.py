"""
session_api.py — Build 1 + Build 3 HTTP surface: /api/v1

The browser fires these best-effort while a kid plays. Every write goes to the
Supabase `sessions` table via supabase_client. NOTHING here may ever affect the
kid: handlers swallow their own failures and always return {"ok": ...} 200 so a
broken backend produces no client-side error.

PRIVACY: country is resolved from the request IP and the IP is then DISCARDED —
never stored, never logged. No names/emails/IP columns (a name may only ever
appear incidentally inside transcript text).
"""

import os
import logging
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import APIRouter, Request, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

import supabase_client as db
import session_analysis

logger = logging.getLogger("nova-sessions")

router = APIRouter(prefix="/api/v1")

_GEOIP = os.getenv("NOVA_GEOIP", "1") != "0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _client_ip(request: Request) -> Optional[str]:
    """Best-effort real client IP (Render sits behind a proxy). Used ONLY to
    resolve a country code, then discarded by the caller."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else None


async def _country_from_request(request: Request) -> Optional[str]:
    """Return a 2-letter country code or None. Prefer edge headers; else one
    guarded geo lookup. The IP is never stored — only the resulting code."""
    for h in ("cf-ipcountry", "x-vercel-ip-country", "x-country-code",
              "fastly-country-code", "x-appengine-country"):
        v = request.headers.get(h)
        if v and len(v) == 2 and v.upper() != "XX":
            return v.upper()
    if not _GEOIP:
        return None
    ip = _client_ip(request)
    if not ip or ip.startswith(("127.", "10.", "192.168.", "::1")):
        return None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(3.0, connect=2.0)) as cx:
            r = await cx.get(f"http://ip-api.com/json/{ip}?fields=status,countryCode")
        if r.status_code == 200:
            d = r.json()
            if d.get("status") == "success":
                return d.get("countryCode")
    except Exception as e:
        logger.debug("[geo] lookup failed (ignored): %s", e)
    return None  # IP goes out of scope here — never persisted


# ── request models ─────────────────────────────────────────────────────────

class StartReq(BaseModel):
    session_id: str
    lang: Optional[str] = None
    device: Optional[str] = None
    app_version: Optional[str] = None
    game: Optional[str] = None  # optional hint; 'mixed' set at end if multiple


class EventsReq(BaseModel):
    session_id: str
    events: Optional[List[Dict[str, Any]]] = None
    transcript: Optional[List[Dict[str, Any]]] = None


class EndReq(BaseModel):
    session_id: str
    stats: Optional[Dict[str, Any]] = None
    game: Optional[str] = None


class FeedbackReq(BaseModel):
    session_id: str
    feedback: Optional[Dict[str, Any]] = None


# ── endpoints ───────────────────────────────────────────────────────────────

@router.post("/session/start")
async def session_start(req: StartReq, request: Request):
    try:
        country = await _country_from_request(request)
        row = {
            "id": req.session_id,
            "started_at": _now_iso(),
            "country": country,
            "lang": req.lang,
            "device": req.device,
            "app_version": req.app_version,
            "game": req.game,
            "events": [],
            "transcript": [],
            "stats": {},
            "feedback": {},
        }
        ok = await db.insert_session(row)
        return {"ok": ok}
    except Exception as e:  # a kid must never see this fail
        logger.warning("[session/start] swallowed: %s", e)
        return {"ok": False}


@router.post("/session/events")
async def session_events(req: EventsReq):
    try:
        ok = await db.append_events(req.session_id, req.events, req.transcript)
        return {"ok": ok}
    except Exception as e:
        logger.warning("[session/events] swallowed: %s", e)
        return {"ok": False}


@router.post("/session/end")
async def session_end(req: EndReq, background: BackgroundTasks):
    try:
        stats = req.stats or {}
        if req.game:
            stats.setdefault("game", req.game)
        ok = await db.end_session(req.session_id, _now_iso(), stats)
        # Build 3: fire ONE cheap analysis after responding — never blocks /end.
        background.add_task(session_analysis.analyze_session, req.session_id)
        return {"ok": ok}
    except Exception as e:
        logger.warning("[session/end] swallowed: %s", e)
        return {"ok": False}


@router.post("/session/feedback")
async def session_feedback(req: FeedbackReq):
    try:
        ok = await db.merge_feedback(req.session_id, req.feedback or {})
        return {"ok": ok}
    except Exception as e:
        logger.warning("[session/feedback] swallowed: %s", e)
        return {"ok": False}


# ── daily rollup (Build 3) — call once/day from an external scheduler ────────
# Render can't run reliable in-process cron on the free/standard flow, so this
# is an endpoint. Point a daily ping (Render Cron Job / cron-job.org / GH Action)
# at it. Protected by X-Rollup-Secret when NOVA_ROLLUP_SECRET is set.

@router.post("/rollup/daily")
async def rollup_daily(request: Request, day: Optional[str] = None):
    secret = os.getenv("NOVA_ROLLUP_SECRET")
    if secret and request.headers.get("x-rollup-secret") != secret:
        return {"ok": False, "error": "unauthorized"}
    try:
        if day:
            d0 = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            d0 = (datetime.now(timezone.utc) - timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0)
        d1 = d0 + timedelta(days=1)
        label = d0.strftime("%Y-%m-%d")
        sessions = await db.sessions_in_range(d0.isoformat(), d1.isoformat())
        sessions = [s for s in sessions if s.get("game") != "_daily"]
        if not sessions:
            return {"ok": True, "day": label, "sessions": 0, "note": "nothing to roll up"}
        text = await session_analysis.rollup_day(label, sessions)
        # Deterministic id per day so re-runs replace the same rollup row.
        import uuid
        rid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"nova-daily-{label}"))
        finished = sum(1 for s in sessions if s.get("ended_at"))
        row = {
            "id": rid,
            "started_at": d0.isoformat(),
            "ended_at": d1.isoformat(),
            "game": "_daily",
            "app_version": "rollup",
            "analysis": text,
            "stats": {"sessions": len(sessions), "finished": finished, "day": label},
        }
        await db.insert_daily_rollup(row)
        return {"ok": True, "day": label, "sessions": len(sessions)}
    except Exception as e:
        logger.warning("[rollup/daily] failed: %s", e)
        return {"ok": False, "error": str(e)}
