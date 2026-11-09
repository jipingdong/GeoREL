"""Small filesystem-backed concurrency slots shared by worker processes."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path


def _validated_prefix(prefix: str) -> str:
    if not prefix or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in prefix):
        raise ValueError("slot prefix may contain only letters, numbers, '-' and '_'")
    return prefix


def cleanup_stale_slots(
    slot_dir: os.PathLike[str] | str,
    prefix: str,
    stale_seconds: float,
) -> int:
    directory = Path(slot_dir)
    if not directory.is_dir():
        return 0
    safe_prefix = _validated_prefix(prefix)
    now = time.time()
    removed = 0
    for path in directory.glob(f"{safe_prefix}_*.slot"):
        try:
            if now - path.stat().st_mtime <= max(0.0, stale_seconds):
                continue
            path.unlink()
            removed += 1
        except FileNotFoundError:
            continue
        except OSError:
            continue
    return removed


def acquire_slot(
    slot_dir: os.PathLike[str] | str,
    prefix: str,
    limit: int,
    timeout: float,
    stale_seconds: float,
    poll_interval: float = 0.1,
) -> str | None:
    directory = Path(slot_dir)
    directory.mkdir(parents=True, exist_ok=True)
    safe_prefix = _validated_prefix(prefix)
    slot_limit = max(1, int(limit))
    deadline = time.monotonic() + max(0.0, float(timeout))

    while True:
        cleanup_stale_slots(directory, safe_prefix, stale_seconds)
        for index in range(slot_limit):
            path = directory / f"{safe_prefix}_{index:04d}.slot"
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            try:
                payload = (
                    f"pid={os.getpid()}\n"
                    f"created_at={datetime.now(timezone.utc).isoformat()}\n"
                ).encode("utf-8")
                os.write(descriptor, payload)
            except Exception:
                os.close(descriptor)
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                raise
            else:
                os.close(descriptor)
                return str(path)

        if time.monotonic() >= deadline:
            return None
        time.sleep(max(0.01, float(poll_interval)))


def release_slot(slot_path: os.PathLike[str] | str | None) -> None:
    if not slot_path:
        return
    try:
        Path(slot_path).unlink()
    except FileNotFoundError:
        pass
