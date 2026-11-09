"""SQLite-backed Wayback snapshot cache and lookup helpers."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from data_crawl.utils.paths import normalize_url


VALID_STATUSES = frozenset({"hit", "no_snapshot", "error"})


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime:
    result = _utc_now() if value is None else value
    if result.tzinfo is None:
        return result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


@dataclass(frozen=True)
class CacheEntry:
    source_url: str
    status: str
    archive_url: str
    checked_at: datetime
    error: str


class WaybackCache:
    def __init__(
        self,
        path: str | Path,
        hit_ttl_seconds: float,
        miss_ttl_seconds: float,
        error_ttl_seconds: float,
    ) -> None:
        self.path = str(Path(path))
        self.ttls = {
            "hit": max(0.0, float(hit_ttl_seconds)),
            "no_snapshot": max(0.0, float(miss_ttl_seconds)),
            "error": max(0.0, float(error_ttl_seconds)),
        }
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS wayback_cache (
                    source_url TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    archive_url TEXT NOT NULL DEFAULT '',
                    checked_at TEXT NOT NULL,
                    error TEXT NOT NULL DEFAULT ''
                )
                """
            )

    def get(self, source_url: object, *, now: datetime | None = None) -> CacheEntry | None:
        url = normalize_url(source_url)
        if not url:
            return None
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT source_url, status, archive_url, checked_at, error
                FROM wayback_cache
                WHERE source_url = ?
                """,
                (url,),
            ).fetchone()
        if row is None or row[1] not in VALID_STATUSES:
            return None
        try:
            checked_at = _as_utc(datetime.fromisoformat(row[3]))
        except (TypeError, ValueError):
            return None
        age_seconds = (_as_utc(now) - checked_at).total_seconds()
        if age_seconds >= self.ttls[row[1]]:
            return None
        return CacheEntry(
            source_url=row[0],
            status=row[1],
            archive_url=row[2] or "",
            checked_at=checked_at,
            error=row[4] or "",
        )

    def put(
        self,
        source_url: object,
        status: str,
        archive_url: str = "",
        *,
        error: str = "",
        now: datetime | None = None,
    ) -> None:
        url = normalize_url(source_url)
        if not url:
            raise ValueError("source_url must not be empty")
        if status not in VALID_STATUSES:
            raise ValueError(f"unsupported Wayback cache status: {status}")
        checked_at = _as_utc(now).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO wayback_cache (
                    source_url, status, archive_url, checked_at, error
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source_url) DO UPDATE SET
                    status = excluded.status,
                    archive_url = excluded.archive_url,
                    checked_at = excluded.checked_at,
                    error = excluded.error
                """,
                (url, status, archive_url or "", checked_at, error or ""),
            )

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM wayback_cache").fetchone()
        return int(row[0])


async def get_wayback_snapshot(
    source_url: object,
    cache: WaybackCache,
    lookup: Callable[[str], Awaitable[str]],
    *,
    now: datetime | None = None,
) -> str:
    url = normalize_url(source_url)
    if not url:
        return ""

    try:
        entry = cache.get(url, now=now)
    except Exception as exc:
        logging.warning("Wayback cache read failed for %s: %s", url, exc.__class__.__name__)
        entry = None
    if entry is not None:
        return entry.archive_url if entry.status == "hit" else ""

    try:
        archive_url = normalize_url(await lookup(url))
    except Exception as exc:
        try:
            cache.put(
                url,
                "error",
                error=f"{exc.__class__.__name__}: {exc}",
                now=now,
            )
        except Exception as cache_exc:
            logging.warning(
                "Wayback cache write failed for %s: %s",
                url,
                cache_exc.__class__.__name__,
            )
        return ""

    status = "hit" if archive_url else "no_snapshot"
    try:
        cache.put(url, status, archive_url, now=now)
    except Exception as exc:
        logging.warning("Wayback cache write failed for %s: %s", url, exc.__class__.__name__)
    return archive_url
