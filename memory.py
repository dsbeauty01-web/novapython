"""
Nova memory store — Day 3 in-memory implementation.

Stores per-kid data across sessions WITHIN a server uptime.
Restarts wipe everything (you chose "no Postgres for now").

Keyed by a stable kidId (browser localStorage). If we ever add Postgres,
this is the only file that changes.
"""

from typing import Optional, List
from dataclasses import dataclass, field, asdict
from threading import RLock
import time


@dataclass
class KidMemory:
    """Everything Nova remembers about one kid."""
    kid_id: str
    name: Optional[str] = None
    age: Optional[int] = None
    total_sessions: int = 0
    max_streak: int = 0
    favorite_move: Optional[str] = None
    best_moments: List[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)


class MemoryStore:
    """Thread-safe in-memory key-value store. Drop-in replaceable with Postgres."""

    def __init__(self):
        self._lock = RLock()
        self._data: dict[str, KidMemory] = {}

    def get(self, kid_id: str) -> KidMemory:
        with self._lock:
            if kid_id not in self._data:
                self._data[kid_id] = KidMemory(kid_id=kid_id)
            return self._data[kid_id]

    def update(self, kid_id: str, **fields) -> KidMemory:
        with self._lock:
            mem = self.get(kid_id)
            for key, value in fields.items():
                if hasattr(mem, key):
                    setattr(mem, key, value)
            mem.last_seen = time.time()
            return mem

    def add_moment(self, kid_id: str, moment: str, keep_last: int = 5):
        with self._lock:
            mem = self.get(kid_id)
            mem.best_moments.append(moment)
            if len(mem.best_moments) > keep_last:
                mem.best_moments = mem.best_moments[-keep_last:]

    def increment_sessions(self, kid_id: str):
        with self._lock:
            mem = self.get(kid_id)
            mem.total_sessions += 1
            mem.last_seen = time.time()

    def record_streak(self, kid_id: str, streak: int):
        with self._lock:
            mem = self.get(kid_id)
            if streak > mem.max_streak:
                mem.max_streak = streak

    def all_kids(self) -> List[KidMemory]:
        with self._lock:
            return list(self._data.values())

    def to_dict(self, kid_id: str) -> dict:
        return asdict(self.get(kid_id))


# Module-level singleton (process-wide)
store = MemoryStore()
