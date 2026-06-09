"""
Nova memory store — Postgres if configured, in-RAM fallback otherwise.

KEY DESIGN: Same API regardless of backend. Calling code never knows or cares.
If DATABASE_URL env var is set on Render, persistent Postgres is used.
Otherwise, in-RAM (fine for dev + first few sessions).

Schema:
    kids:
      kid_id PK
      name, age
      total_sessions, max_streak
      favorite_move, favorite_song
      best_moments JSONB (list of strings)
      energy_read (shy|hyped|quiet|unknown)
      shared_facts JSONB (dict: things kid mentioned — pet, sibling, color, etc.)
      message_history JSONB (last 12 chat turns)
      languages_seen JSONB (list)
      created_at, last_seen

This is the long-term brain. Loaded on session start, saved on goodbye + on
notable moments mid-session.
"""

import os
import json
import time
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from threading import RLock

logger = logging.getLogger("nova-v207-memory")


@dataclass
class KidMemory:
    """Everything Nova remembers about one kid."""
    kid_id: str
    name: Optional[str] = None
    age: Optional[int] = None
    total_sessions: int = 0
    max_streak: int = 0
    favorite_move: Optional[str] = None
    favorite_song: Optional[str] = None
    best_moments: List[str] = field(default_factory=list)
    energy_read: str = "unknown"  # shy | hyped | quiet | balanced
    shared_facts: Dict[str, str] = field(default_factory=dict)  # {"pet": "Mango (cat)", "color": "yellow"}
    message_history: List[Dict[str, str]] = field(default_factory=list)  # [{"role":"user/assistant","text":"..."}]
    languages_seen: List[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    def add_message(self, role: str, text: str, keep_last: int = 12):
        self.message_history.append({"role": role, "text": text, "t": time.time()})
        if len(self.message_history) > keep_last:
            self.message_history = self.message_history[-keep_last:]

    def add_moment(self, moment: str, keep_last: int = 8):
        if not moment or moment in self.best_moments:
            return
        self.best_moments.append(moment)
        if len(self.best_moments) > keep_last:
            self.best_moments = self.best_moments[-keep_last:]

    def add_shared_fact(self, key: str, value: str):
        self.shared_facts[key.lower().strip()] = value.strip()


# ════════════════════════════════════════════════════════════════
# IN-RAM STORE (default, no Postgres needed)
# ════════════════════════════════════════════════════════════════
class _RAMStore:
    def __init__(self):
        self._lock = RLock()
        self._data: Dict[str, KidMemory] = {}

    def get(self, kid_id: str) -> KidMemory:
        with self._lock:
            if kid_id not in self._data:
                self._data[kid_id] = KidMemory(kid_id=kid_id)
            return self._data[kid_id]

    def save(self, mem: KidMemory):
        with self._lock:
            mem.last_seen = time.time()
            self._data[mem.kid_id] = mem

    def all_kids(self) -> list:
        with self._lock:
            return list(self._data.keys())


# ════════════════════════════════════════════════════════════════
# POSTGRES STORE (used if DATABASE_URL env var is set)
# Postgres is lazy-imported so the worker boots fine without it.
# ════════════════════════════════════════════════════════════════
class _PostgresStore:
    def __init__(self, database_url: str):
        import psycopg2
        from psycopg2.extras import Json, RealDictCursor
        self._psycopg2 = psycopg2
        self._Json = Json
        self._RealDictCursor = RealDictCursor
        self._url = database_url
        self._ensure_schema()
        logger.info("[memory] Postgres backend active")

    def _conn(self):
        return self._psycopg2.connect(self._url)

    def _ensure_schema(self):
        with self._conn() as c, c.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kids (
                    kid_id TEXT PRIMARY KEY,
                    name TEXT,
                    age INT,
                    total_sessions INT DEFAULT 0,
                    max_streak INT DEFAULT 0,
                    favorite_move TEXT,
                    favorite_song TEXT,
                    best_moments JSONB DEFAULT '[]',
                    energy_read TEXT DEFAULT 'unknown',
                    shared_facts JSONB DEFAULT '{}',
                    message_history JSONB DEFAULT '[]',
                    languages_seen JSONB DEFAULT '[]',
                    created_at DOUBLE PRECISION,
                    last_seen DOUBLE PRECISION
                );
            """)
            c.commit()

    def get(self, kid_id: str) -> KidMemory:
        with self._conn() as c, c.cursor(cursor_factory=self._RealDictCursor) as cur:
            cur.execute("SELECT * FROM kids WHERE kid_id = %s", (kid_id,))
            row = cur.fetchone()
            if not row:
                return KidMemory(kid_id=kid_id)
            return KidMemory(
                kid_id=row["kid_id"],
                name=row["name"],
                age=row["age"],
                total_sessions=row["total_sessions"] or 0,
                max_streak=row["max_streak"] or 0,
                favorite_move=row["favorite_move"],
                favorite_song=row["favorite_song"],
                best_moments=list(row["best_moments"] or []),
                energy_read=row["energy_read"] or "unknown",
                shared_facts=dict(row["shared_facts"] or {}),
                message_history=list(row["message_history"] or []),
                languages_seen=list(row["languages_seen"] or []),
                created_at=row["created_at"] or time.time(),
                last_seen=row["last_seen"] or time.time(),
            )

    def save(self, mem: KidMemory):
        mem.last_seen = time.time()
        with self._conn() as c, c.cursor() as cur:
            cur.execute("""
                INSERT INTO kids (kid_id, name, age, total_sessions, max_streak,
                                  favorite_move, favorite_song, best_moments,
                                  energy_read, shared_facts, message_history,
                                  languages_seen, created_at, last_seen)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (kid_id) DO UPDATE SET
                    name=EXCLUDED.name,
                    age=EXCLUDED.age,
                    total_sessions=EXCLUDED.total_sessions,
                    max_streak=EXCLUDED.max_streak,
                    favorite_move=EXCLUDED.favorite_move,
                    favorite_song=EXCLUDED.favorite_song,
                    best_moments=EXCLUDED.best_moments,
                    energy_read=EXCLUDED.energy_read,
                    shared_facts=EXCLUDED.shared_facts,
                    message_history=EXCLUDED.message_history,
                    languages_seen=EXCLUDED.languages_seen,
                    last_seen=EXCLUDED.last_seen;
            """, (
                mem.kid_id, mem.name, mem.age, mem.total_sessions, mem.max_streak,
                mem.favorite_move, mem.favorite_song,
                self._Json(mem.best_moments), mem.energy_read,
                self._Json(mem.shared_facts), self._Json(mem.message_history),
                self._Json(mem.languages_seen),
                mem.created_at, mem.last_seen,
            ))
            c.commit()

    def all_kids(self) -> list:
        with self._conn() as c, c.cursor() as cur:
            cur.execute("SELECT kid_id FROM kids")
            return [r[0] for r in cur.fetchall()]


# ════════════════════════════════════════════════════════════════
# PUBLIC API — uniform regardless of backend
# ════════════════════════════════════════════════════════════════
class MemoryStore:
    """Thread-safe per-kid memory. Postgres if available, RAM otherwise."""

    def __init__(self):
        self._lock = RLock()
        self._backend = self._make_backend()

    def _make_backend(self):
        url = os.getenv("DATABASE_URL", "").strip()
        if url:
            try:
                return _PostgresStore(url)
            except Exception as e:
                logger.warning(f"[memory] Postgres init failed → using RAM. Err: {e}")
        logger.info("[memory] RAM backend active (set DATABASE_URL for persistence)")
        return _RAMStore()

    def get(self, kid_id: str) -> KidMemory:
        return self._backend.get(kid_id)

    def save(self, mem: KidMemory):
        return self._backend.save(mem)

    def update(self, kid_id: str, **fields) -> KidMemory:
        with self._lock:
            mem = self.get(kid_id)
            for key, value in fields.items():
                if hasattr(mem, key):
                    setattr(mem, key, value)
            self.save(mem)
            return mem

    def add_moment(self, kid_id: str, moment: str):
        with self._lock:
            mem = self.get(kid_id)
            mem.add_moment(moment)
            self.save(mem)

    def add_shared_fact(self, kid_id: str, key: str, value: str):
        with self._lock:
            mem = self.get(kid_id)
            mem.add_shared_fact(key, value)
            self.save(mem)

    def add_message(self, kid_id: str, role: str, text: str):
        with self._lock:
            mem = self.get(kid_id)
            mem.add_message(role, text)
            self.save(mem)

    def increment_sessions(self, kid_id: str):
        with self._lock:
            mem = self.get(kid_id)
            mem.total_sessions += 1
            self.save(mem)

    def record_streak(self, kid_id: str, streak: int):
        with self._lock:
            mem = self.get(kid_id)
            if streak > mem.max_streak:
                mem.max_streak = streak
                self.save(mem)

    def all_kids(self) -> list:
        with self._lock:
            return self._backend.all_kids()


# Module singleton
store = MemoryStore()
