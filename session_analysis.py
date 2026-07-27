"""
session_analysis.py — Build 3: one cheap LLM read of each finished session.

Fired from /api/v1/session/end as a background task. Reads the stored row,
sends events + transcript + stats to a small model, writes a <=10-line
observation back to the `analysis` column.

GUARDRAIL: product observations only. The prompt forbids any speculation about
the child's home, health, family, or wellbeing — we describe the SESSION, not
the kid's life. Best-effort: any failure logs and leaves `analysis` empty.
"""

import os
import json
import logging

import httpx

import supabase_client as db

logger = logging.getLogger("nova-sessions")

_MODEL = os.getenv("NOVA_ANALYSIS_MODEL", "gpt-4o-mini")
_TIMEOUT = httpx.Timeout(40.0, connect=6.0)

_SYSTEM = (
    "You analyze ONE session of a young child (ages 4-7) playing a browser dance "
    "game with an AI dance teacher named Nova. You are given the session's event "
    "log, speech transcript, and stats. Produce a terse product report, AT MOST 10 "
    "lines, EXACTLY these labels, one line each:\n"
    "ENGAGEMENT: finished? talked back? replayed? when/if they left\n"
    "BEST MOMENT: the single highlight, with its timestamp\n"
    "STRUGGLE: where they lost it / went quiet / detection missed\n"
    "CUE HEALTH: did Nova's lines land on time (judge from event timestamps)\n"
    "FEEDBACK: the face they tapped + what they said, interpreted\n"
    "ONE FIX: the single most valuable product improvement this session suggests\n"
    "Rules: observe the SESSION only. NEVER speculate about the child's home, "
    "health, family, mood, or wellbeing. If data is missing for a line, write "
    "'n/a'. No preamble, no markdown, just the six labelled lines."
)


def _trim(obj, limit=12000):
    """Keep the LLM input cheap and bounded."""
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "…(truncated)"


async def analyze_session(session_id: str) -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        logger.info("[analysis] OPENAI_API_KEY not set — skipping analysis for %s", session_id)
        return None

    row = await db.get_session(session_id)
    if not row:
        logger.warning("[analysis] no row for %s — skipping", session_id)
        return None

    user = (
        f"GAME: {row.get('game')}\n"
        f"LANG: {row.get('lang')}  DEVICE: {row.get('device')}  COUNTRY: {row.get('country')}\n"
        f"STARTED: {row.get('started_at')}  ENDED: {row.get('ended_at')}\n"
        f"STATS: {_trim(row.get('stats'), 2000)}\n"
        f"FEEDBACK: {_trim(row.get('feedback'), 1000)}\n"
        f"TRANSCRIPT: {_trim(row.get('transcript'), 5000)}\n"
        f"EVENTS: {_trim(row.get('events'), 6000)}\n"
    )

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as cx:
            r = await cx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": _MODEL,
                    "temperature": 0.2,
                    "max_tokens": 400,
                    "messages": [
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": user},
                    ],
                },
            )
        if r.status_code != 200:
            logger.warning("[analysis] openai %s: %s", r.status_code, r.text[:200])
            return None
        text = r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning("[analysis] LLM call failed for %s: %s", session_id, e)
        return None

    await db.set_analysis(session_id, text)
    logger.info("[analysis] wrote %d chars for %s", len(text), session_id)
    return text


_ROLLUP_SYSTEM = (
    "You are given the per-session analyses from one day of a kids' dance app. "
    "Write a single terse daily rollup, AT MOST 12 lines: total sessions, how "
    "many finished, the standout moment of the day, the most common struggle, "
    "cue-health trend, feedback-face distribution, and the ONE fix to prioritize "
    "tomorrow. Product observations only — never speculate about any child's life."
)


async def rollup_day(day_label: str, sessions: list) -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if not key or not sessions:
        return None
    payload = _trim(
        [
            {
                "game": s.get("game"),
                "analysis": s.get("analysis"),
                "feedback": s.get("feedback"),
                "stats": s.get("stats"),
                "country": s.get("country"),
                "lang": s.get("lang"),
            }
            for s in sessions
        ],
        limit=20000,
    )
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as cx:
            r = await cx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": _MODEL,
                    "temperature": 0.2,
                    "max_tokens": 500,
                    "messages": [
                        {"role": "system", "content": _ROLLUP_SYSTEM},
                        {"role": "user", "content": f"DAY {day_label}, {len(sessions)} sessions:\n{payload}"},
                    ],
                },
            )
        if r.status_code != 200:
            logger.warning("[rollup] openai %s: %s", r.status_code, r.text[:200])
            return None
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning("[rollup] failed: %s", e)
        return None
