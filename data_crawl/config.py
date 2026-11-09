"""Command-line parsing and immutable runtime configuration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def parse_bool_arg(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Use True or False.")


def parse_datetime_value(value: object, *, is_end: bool = False) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    date_only = False
    for fmt in ("%Y%m%d%H%M%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            date_only = fmt in {"%Y%m%d", "%Y-%m-%d"}
            break
        except ValueError:
            continue
    else:
        raise ValueError(
            "Unsupported datetime format. Use YYYYMMDDHHMMSS, YYYYMMDD, "
            "YYYY-MM-DD, or YYYY-MM-DD HH:MM:SS."
        )

    if is_end and date_only:
        return parsed.replace(hour=23, minute=59, second=59)
    return parsed


@dataclass(frozen=True)
class RuntimeConfig:
    data_dir: str
    year: int
    zip_selection_mode: str
    retry_csv_path: str
    since_datetime: datetime
    end_datetime: datetime | None
    state_file: str
    saved_files_path: str
    no_progress: bool
    proxy: str
    zip_workers: int
    url_concurrency: int
    async_batch_size: int
    download_strategy: str
    wayback_mode: str
    wayback_concurrency: int
    wayback_global_concurrency: int
    wayback_cache_path: str
    wayback_hit_ttl_seconds: float
    wayback_miss_ttl_seconds: float
    wayback_error_ttl_seconds: float
    wayback_slot_dir: str
    wayback_lock_timeout: float
    wayback_lock_stale_seconds: float
    selenium_mode: str
    selenium_max_concurrency: int
    selenium_slot_dir: str
    selenium_lock_timeout: float
    chromedriver_path: str
    chrome_binary_path: str
    log_dir: str
    log_max_bytes: int
    log_backup_count: int
    filter_country_coded: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_data_dir = str(Path(__file__).resolve().parent.parent / "data")
    parser = argparse.ArgumentParser(
        description="Fetch webpage titles for GDELT SOURCEURL records."
    )
    parser.add_argument("--data-dir", default=default_data_dir)
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument(
        "--zip-selection-mode",
        choices=["default", "retry_csv", "since_datetime", "incremental"],
        default="since_datetime",
    )
    parser.add_argument("--retry-csv-path", default="")
    parser.add_argument("--since-datetime", default="20260801000000")
    parser.add_argument("--end-datetime", default="20260801234500")
    parser.add_argument("--state-file", default="")
    parser.add_argument(
        "--filter-country-coded",
        type=parse_bool_arg,
        default=False,
        metavar="True|False",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--config-check", action="store_true")
    parser.add_argument("--zip-workers", type=int, default=4)
    parser.add_argument(
        "--url-concurrency", type=int, default=100
    )
    parser.add_argument(
        "--download-strategy",
        choices=["direct_first", "wayback_first"],
        default="direct_first",
    )
    parser.add_argument("--proxy", default="http://127.0.0.1:7900")
    parser.add_argument(
        "--wayback-mode",
        choices=["on", "off"],
        default="on",
    )
    parser.add_argument(
        "--wayback-concurrency",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--wayback-global-concurrency",
        type=int,
        default=60,
    )
    parser.add_argument("--wayback-cache-path", default="")
    parser.add_argument("--wayback-hit-ttl-hours", type=float, default=720.0)
    parser.add_argument("--wayback-miss-ttl-hours", type=float, default=24.0)
    parser.add_argument("--wayback-error-ttl-minutes", type=float, default=15.0)
    parser.add_argument(
        "--selenium-mode",
        choices=["on", "off"],
        default="on",
    )
    parser.add_argument(
        "--selenium-max-concurrency",
        type=int,
        default=30,
    )
    return parser.parse_args(argv)


def _data_relative_path(data_dir: Path, value: object, default_relative: str) -> str:
    text = str(value or "").strip()
    path = Path(text) if text else Path(default_relative)
    if not path.is_absolute():
        path = data_dir / path
    return str(path.resolve())


def resolve_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    data_dir = Path(str(args.data_dir)).expanduser().resolve()
    since_datetime = parse_datetime_value(args.since_datetime)
    if since_datetime is None:
        raise ValueError("since-datetime must not be empty")
    end_datetime = parse_datetime_value(args.end_datetime, is_end=True)
    if end_datetime is not None and end_datetime < since_datetime:
        raise ValueError("end-datetime must be greater than or equal to since-datetime")

    module_root = Path(__file__).resolve().parent
    chromedriver = str(module_root / "chromedriver.exe")
    chrome_binary = ""

    return RuntimeConfig(
        data_dir=str(data_dir),
        year=int(args.year),
        zip_selection_mode=str(args.zip_selection_mode),
        retry_csv_path=_data_relative_path(data_dir, args.retry_csv_path, "crawl_output/stats/empty_stats_retry.csv"),
        since_datetime=since_datetime,
        end_datetime=end_datetime,
        state_file=_data_relative_path(data_dir, args.state_file, "crawl_output/state/fulltext_last_processed.txt"),
        saved_files_path=str((data_dir / "crawl_output" / "saved_files.txt").resolve()),
        no_progress=bool(args.no_progress),
        proxy=str(args.proxy or "").strip(),
        zip_workers=max(1, int(args.zip_workers)),
        url_concurrency=max(1, int(args.url_concurrency)),
        async_batch_size=200,
        download_strategy=str(args.download_strategy),
        wayback_mode=str(args.wayback_mode),
        wayback_concurrency=max(1, int(args.wayback_concurrency)),
        wayback_global_concurrency=max(1, int(args.wayback_global_concurrency)),
        wayback_cache_path=_data_relative_path(
            data_dir, args.wayback_cache_path, "crawl_output/state/wayback_cache.sqlite3"
        ),
        wayback_hit_ttl_seconds=max(0.0, float(args.wayback_hit_ttl_hours) * 3600),
        wayback_miss_ttl_seconds=max(0.0, float(args.wayback_miss_ttl_hours) * 3600),
        wayback_error_ttl_seconds=max(0.0, float(args.wayback_error_ttl_minutes) * 60),
        wayback_slot_dir=_data_relative_path(data_dir, "", "crawl_output/state/wayback_slots"),
        wayback_lock_timeout=120.0,
        wayback_lock_stale_seconds=900.0,
        selenium_mode=str(args.selenium_mode),
        selenium_max_concurrency=max(1, int(args.selenium_max_concurrency)),
        selenium_slot_dir=_data_relative_path(data_dir, "", "crawl_output/state/selenium_slots"),
        selenium_lock_timeout=120.0,
        chromedriver_path=str(Path(chromedriver).expanduser().resolve()),
        chrome_binary_path=(
            str(Path(chrome_binary).expanduser().resolve()) if chrome_binary else ""
        ),
        log_dir=_data_relative_path(data_dir, "", "crawl_output/logs"),
        log_max_bytes=100 * 1024 * 1024,
        log_backup_count=5,
        filter_country_coded=bool(args.filter_country_coded),
    )
