"""Explicit, per-process logging configuration for the crawler."""

from __future__ import annotations

import logging
import os
import sys
from contextlib import suppress
from logging.handlers import RotatingFileHandler
from pathlib import Path

from data_crawl.config import RuntimeConfig


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


def silence_native_stderr() -> None:
    """Discard native stderr inherited by browser subprocesses."""
    with suppress(Exception):
        sys.stderr.flush()
    null_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_fd, 2)
    finally:
        os.close(null_fd)
    if os.name == "nt":
        with suppress(Exception):
            import ctypes

            ctypes.windll.kernel32.FreeConsole()


def build_rotating_handler(
    log_path: os.PathLike[str] | str,
    max_bytes: int,
    backup_count: int,
) -> RotatingFileHandler:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        path,
        maxBytes=max(1, int(max_bytes)),
        backupCount=max(0, int(backup_count)),
        encoding="utf-8",
        delay=False,
    )
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return handler


def _reset_root_handlers() -> logging.Logger:
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    root.setLevel(logging.INFO)
    logging.raiseExceptions = False
    return root


def setup_main_logging(config: RuntimeConfig) -> None:
    root = _reset_root_handlers()
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(stream)
    root.addHandler(
        build_rotating_handler(
            Path(config.log_dir) / "fulltext.log",
            config.log_max_bytes,
            config.log_backup_count,
        )
    )


def setup_worker_logging(config: RuntimeConfig, pid: int | None = None) -> None:
    root = _reset_root_handlers()
    worker_pid = os.getpid() if pid is None else int(pid)
    root.addHandler(
        build_rotating_handler(
            Path(config.log_dir) / f"fulltext_worker_{worker_pid}.log",
            config.log_max_bytes,
            config.log_backup_count,
        )
    )
