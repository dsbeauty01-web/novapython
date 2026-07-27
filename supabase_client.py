"""
supabase_client.py — thin, dependency-light data layer for the sessions table.

Talks to Supabase PostgREST directly over httpx (already a worker dependency);
we deliberately DON'T pull in the heavy `supabase` SDK.

HARD LAW (secrets): SUPABASE_URL + SUPABASE_SERVICE_KEY come ONLY from the
worker environment (Render env / .env). They never touch a client file and are
never returned to the browser. The browser talks only to the worker.

HARD LAW (gameplay): a dead or misconfigured Supabase must NEVER affect the kid.
Every call here is best-effort — on any failure it logs and returns False/None.
If the env is not set the whole layer no-ops so the worker still boots and plays.
"""

import os
import logging

import httpx

logger = logging.getLogger("nova-sessions")

SUPABASE_URL = (os.getenv("SUPABASE_URL", "") or "").rstrip("/")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "") or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

_TIMEOUT = httpx.Timeout(8.0, connect=4.0)


def configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)


def _headers(extra: dict | None = None) -> dict:
    h = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
    }
    if extra:
        h.update(extra)
    return h


async def _rest(method: str, path: str, *, params=None, json=None, prefer=None):
    """One best-effort PostgREST call. Returns (ok, data). Never raises."""
    if not configured():
        logger.debug("[supabase] skipped %s %s (not configured)", method, path)
        return False, None
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    headers = _headers({"Prefer": prefer} if prefer else None)
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as cx:
            r = await cx.request(method, url, headers=headers, params=params, json=json)
        if r.status_code >= 300:
            logger.warning("[supabase] %s %s -> %s %s", method, path, r.status_code, r.text[:200])
            return False, None
        body = None
        if r.content:
            try:
                body = r.json()
            except Exception:
                body = None
        return True, body
    except Exception as e:
        logger.warning("[supabase] %s %s FAILED: %s", method, path, e)
        return False, None


# ── writes used by the session router ──────────────────────────────────────

async def insert_session(row: dict) -> bool:
    """Insert one session row. Idempotent-ish: id is the PK, a duplicate /start
    (double-fire from the client) is swallowed as a conflict, not an error."""
    ok, _ = await _rest(
        "POST", "sessions",
        json=row,
        prefer="return=minimal,resolution=ignore-duplicates",
    )
    return ok


async def append_events(session_id: str, events: list | None, transcript: list | None) -> bool:
    """Atomically append batches to the events/transcript jsonb arrays via RPC.
    The client sends only NEW items since its last batch (deltas, no retry), so
    the append is naturally non-duplicating without server-side dedup."""
    ok, _ = await _rest(
        "POST", "rpc/append_session_events",
        json={
            "p_id": session_id,
            "p_events": events or [],
            "p_transcript": transcript or [],
        },
        prefer="return=minimal",
    )
    return ok


async def end_session(session_id: str, ended_at: str, stats: dict | None) -> bool:
    ok, _ = await _rest(
        "PATCH", "sessions",
        params={"id": f"eq.{session_id}"},
        json={"ended_at": ended_at, "stats": stats or {}},
        prefer="return=minimal",
    )
    return ok


async def merge_feedback(session_id: str, feedback: dict) -> bool:
    """Merge (shallow) into the feedback jsonb via RPC so partial feedback
    posts (face now, spoken line a moment later) accumulate instead of clobber."""
    ok, _ = await _rest(
        "POST", "rpc/merge_session_feedback",
        json={"p_id": session_id, "p_feedback": feedback or {}},
        prefer="return=minimal",
    )
    return ok


async def set_analysis(session_id: str, analysis: str) -> bool:
    ok, _ = await _rest(
        "PATCH", "sessions",
        params={"id": f"eq.{session_id}"},
        json={"analysis": analysis},
        prefer="return=minimal",
    )
    return ok


async def get_session(session_id: str) -> dict | None:
    """Read one row back (used by the analysis step to assemble LLM input)."""
    ok, data = await _rest(
        "GET", "sessions",
        params={"id": f"eq.{session_id}", "select": "*"},
    )
    if ok and isinstance(data, list) and data:
        return data[0]
    return None


async def sessions_in_range(start_iso: str, end_iso: str) -> list:
    """All sessions that ended within [start, end) — for the daily rollup."""
    ok, data = await _rest(
        "GET", "sessions",
        params={
            "select": "id,game,stats,analysis,feedback,started_at,ended_at,country,lang",
            "ended_at": f"gte.{start_iso}",
            "and": f"(ended_at.lt.{end_iso})",
        },
    )
    return data if (ok and isinstance(data, list)) else []


async def insert_daily_rollup(row: dict) -> bool:
    """Upsert on id (deterministic per day) so re-running a rollup replaces it."""
    ok, _ = await _rest(
        "POST", "sessions",
        params={"on_conflict": "id"},
        json=row,
        prefer="return=minimal,resolution=merge-duplicates",
    )
    return ok
