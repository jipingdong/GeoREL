"""Command-line orchestration for the full-text crawler."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import logging
import os
import tempfile
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
from datetime import datetime
from typing import Iterable

import pandas as pd
from tqdm.auto import tqdm

from data_crawl.config import RuntimeConfig, parse_args, parse_datetime_value, resolve_runtime_config
from data_crawl.utils.fetchers import fetch_all_async
from data_crawl.utils.locks import cleanup_stale_slots
from data_crawl.utils.logging_utils import setup_main_logging, setup_worker_logging, silence_native_stderr
from data_crawl.utils.paths import (
    extract_zip_datetime_str,
    is_timestamp_zip,
    zip_output_paths,
    zip_root,
)
from data_crawl.utils.storage import (
    ensure_unzipped,
    is_missing_title,
    load_existing_hash_titles,
    load_urls_df,
    restore_titles_from_csv,
    write_dataframe_csv_atomic,
)
from data_crawl.utils.wayback import WaybackCache


EVENT_COLUMNS = [
    "GLOBALEVENTID",
    "SQLDATE",
    "MonthYear",
    "Year",
    "FractionDate",
    "Actor1Code",
    "Actor1Name",
    "Actor1CountryCode",
    "Actor1KnownGroupCode",
    "Actor1EthnicCode",
    "Actor1Religion1Code",
    "Actor1Religion2Code",
    "Actor1Type1Code",
    "Actor1Type2Code",
    "Actor1Type3Code",
    "Actor2Code",
    "Actor2Name",
    "Actor2CountryCode",
    "Actor2KnownGroupCode",
    "Actor2EthnicCode",
    "Actor2Religion1Code",
    "Actor2Religion2Code",
    "Actor2Type1Code",
    "Actor2Type2Code",
    "Actor2Type3Code",
    "IsRootEvent",
    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    "Actor1Geo_Type",
    "Actor1Geo_FullName",
    "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code",
    "Actor1Geo_ADM2Code",
    "Actor1Geo_Lat",
    "Actor1Geo_Long",
    "Actor1Geo_FeatureID",
    "Actor2Geo_Type",
    "Actor2Geo_FullName",
    "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code",
    "Actor2Geo_ADM2Code",
    "Actor2Geo_Lat",
    "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    "ActionGeo_Type",
    "ActionGeo_FullName",
    "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code",
    "ActionGeo_ADM2Code",
    "ActionGeo_Lat",
    "ActionGeo_Long",
    "ActionGeo_FeatureID",
    "DATEADDED",
    "SOURCEURL",
]


TITLE_MISSING_VALUES = ("", "-1")


_WORKER_CONFIG: RuntimeConfig | None = None


def initialize_worker(config: RuntimeConfig) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = config
    setup_worker_logging(config)
    WaybackCache(
        config.wayback_cache_path,
        config.wayback_hit_ttl_seconds,
        config.wayback_miss_ttl_seconds,
        config.wayback_error_ttl_seconds,
    )
    cleanup_stale_slots(
        config.wayback_slot_dir,
        "wayback",
        config.wayback_lock_stale_seconds,
    )
    cleanup_stale_slots(
        config.selenium_slot_dir,
        "selenium",
        config.wayback_lock_stale_seconds,
    )


def initialize_pool_worker(config: RuntimeConfig) -> None:
    silence_native_stderr()
    initialize_worker(config)


def _require_worker_config() -> RuntimeConfig:
    if _WORKER_CONFIG is None:
        raise RuntimeError("worker configuration is not initialized")
    return _WORKER_CONFIG


def process_zip(
    zip_file: str,
    data_dir: str,
    zip_file_path: str,
    *,
    columns_name_events: list[str] | None = None,
) -> tuple[str, int, int, str]:
    config = _require_worker_config()
    logging.info("Processing ZIP %s", zip_file)
    datetime_str = extract_zip_datetime_str(zip_file)
    file_path, csv_path = zip_output_paths(data_dir, zip_file)
    df_path = os.path.join(file_path, f"{datetime_str}.export.CSV")

    if not ensure_unzipped(zip_file_path, file_path, df_path):
        return zip_file, 0, 0, "skipped-empty-zip"

    urls_df = load_urls_df(
        df_path,
        EVENT_COLUMNS if columns_name_events is None else columns_name_events,
        filter_country_coded=config.filter_country_coded,
    )
    urls_df = restore_titles_from_csv(urls_df, csv_path)
    if urls_df["title"].map(is_missing_title).any():
        urls_df = load_existing_hash_titles(urls_df, file_path)
    if urls_df["title"].map(is_missing_title).any():
        urls_df = asyncio.run(fetch_all_async(urls_df, file_path, config))

    write_dataframe_csv_atomic(urls_df, csv_path)
    empty_total = int(urls_df["title"].map(is_missing_title).sum())
    status = "completed" if empty_total == 0 else "incomplete"
    return zip_file, empty_total, len(urls_df), status


def run_zip_pipeline(
    zip_file_paths: Mapping[str, str],
    data_dir: str,
    config: RuntimeConfig,
) -> Iterator[tuple[str, int, int, str]]:
    selected = dict(zip_file_paths)
    if config.zip_workers == 1:
        initialize_worker(config)
        for zip_file, zip_path in selected.items():
            yield process_zip(zip_file, data_dir, zip_path)
        return

    with ProcessPoolExecutor(
        max_workers=config.zip_workers,
        initializer=initialize_pool_worker,
        initargs=(config,),
    ) as executor:
        futures = {
            executor.submit(process_zip, zip_file, data_dir, zip_path): zip_file
            for zip_file, zip_path in selected.items()
        }
        for future in as_completed(futures):
            yield future.result()


def is_file_empty(path: os.PathLike[str] | str) -> bool:
    try:
        return os.path.getsize(path) == 0
    except OSError:
        return True


def parse_zip_datetime_value(value: object, is_end: bool = False) -> datetime | None:
    return parse_datetime_value(value, is_end=is_end)


def collect_zip_files(data_dir: os.PathLike[str] | str, year: object) -> dict[str, str]:
    root = zip_root(data_dir)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Zip directory not found: {root}")

    year_text = str(year) if year else ""
    zip_paths: dict[str, str] = {}
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not is_timestamp_zip(filename):
                continue
            if year_text and not filename.startswith(year_text):
                continue
            path = os.path.join(dirpath, filename)
            if filename in zip_paths:
                logging.warning("Duplicate zip filename ignored: %s", path)
                continue
            zip_paths[filename] = path
    return dict(sorted(zip_paths.items()))


def _atomic_write_text(path: os.PathLike[str] | str, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, part_path = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".part", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(part_path, target)
    finally:
        try:
            os.remove(part_path)
        except FileNotFoundError:
            pass


def load_saved_file_set(saved_files_path: os.PathLike[str] | str) -> set[str]:
    if is_file_empty(saved_files_path):
        return set()
    with open(saved_files_path, "r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def write_saved_file_set(
    saved_files_path: os.PathLike[str] | str,
    saved_files_set: Iterable[str],
) -> None:
    content = "".join(f"{name}\n" for name in sorted(set(saved_files_set)))
    _atomic_write_text(saved_files_path, content)


def count_missing_titles(csv_path: os.PathLike[str] | str) -> int | None:
    if is_file_empty(csv_path):
        return None
    try:
        frame = pd.read_csv(csv_path, usecols=["title"], dtype={"title": "string"})
    except Exception as exc:
        logging.warning("Failed to inspect title cache %s: %s", csv_path, exc.__class__.__name__)
        return None
    if frame.empty:
        return None
    titles = frame["title"].fillna("")
    return int(titles.isin(TITLE_MISSING_VALUES).sum())


def build_zip_worklist(
    zip_files: Iterable[str],
    data_dir: os.PathLike[str] | str,
    saved_files_set: set[str],
) -> tuple[list[str], set[str]]:
    pending: list[str] = []
    updated_saved = set(saved_files_set)
    for zip_file in sorted(zip_files):
        if zip_file not in saved_files_set:
            pending.append(zip_file)
            continue
        _, csv_path = zip_output_paths(data_dir, zip_file)
        missing_total = count_missing_titles(csv_path)
        if missing_total is None or missing_total > 0:
            pending.append(zip_file)
            updated_saved.discard(zip_file)
    return pending, updated_saved


def filter_zip_files_range(
    zip_files: Iterable[str],
    since_datetime: datetime,
    end_datetime: datetime | None = None,
) -> list[str]:
    selected: list[str] = []
    for zip_file in sorted(zip_files):
        try:
            zip_dt = parse_zip_datetime_value(extract_zip_datetime_str(zip_file))
        except ValueError:
            continue
        if zip_dt is None or zip_dt < since_datetime:
            continue
        if end_datetime is not None and zip_dt > end_datetime:
            continue
        selected.append(zip_file)
    return selected


def compute_incremental_watermark(
    zip_files: Iterable[str], result_map: Mapping[str, str]
) -> str:
    watermark = ""
    for zip_file in sorted(zip_files):
        if result_map.get(zip_file) != "completed":
            break
        watermark = zip_file
    return watermark


def load_incremental_watermark(
    state_file: os.PathLike[str] | str,
) -> tuple[str, datetime | None]:
    if is_file_empty(state_file):
        return "", None
    with open(state_file, "r", encoding="utf-8") as handle:
        raw_value = handle.read().strip()
    if not raw_value:
        return "", None
    return raw_value, parse_zip_datetime_value(raw_value)


def write_incremental_watermark(
    state_file: os.PathLike[str] | str, zip_file: str
) -> None:
    watermark = extract_zip_datetime_str(zip_file)
    _atomic_write_text(state_file, f"{watermark}\n")


def load_retry_zip_files(
    retry_csv_path: os.PathLike[str] | str,
    year: object,
    available_zip_files: Iterable[str],
) -> list[str]:
    if is_file_empty(retry_csv_path):
        raise FileNotFoundError(f"Retry CSV not found or empty: {retry_csv_path}")
    try:
        retry_frame = pd.read_csv(
            retry_csv_path,
            usecols=["zip_name"],
            dtype={"zip_name": "string"},
        )
    except ValueError as exc:
        raise ValueError(
            f"Retry CSV is missing required column 'zip_name': {retry_csv_path}"
        ) from exc

    year_text = str(year) if year else ""
    available = set(available_zip_files)
    selected: set[str] = set()
    for raw_name in retry_frame["zip_name"].dropna().astype(str):
        name = raw_name.strip()
        if not name:
            continue
        if not name.lower().endswith(".zip"):
            name = f"{name}.zip"
        if year_text and not name.startswith(year_text):
            continue
        if name in available:
            selected.add(name)
    return sorted(selected)


def select_incremental_candidates(
    zip_files: Iterable[str],
    data_dir: os.PathLike[str] | str,
    saved_files_set: set[str],
    since_datetime: datetime,
    end_datetime: datetime | None,
    watermark_datetime: datetime | None,
) -> tuple[list[str], set[str]]:
    in_range = filter_zip_files_range(zip_files, since_datetime, end_datetime)
    retry_files, updated_saved = build_zip_worklist(in_range, data_dir, saved_files_set)
    newer = [
        name
        for name in in_range
        if watermark_datetime is None
        or parse_zip_datetime_value(extract_zip_datetime_str(name)) > watermark_datetime
    ]
    return sorted(set(newer) | set(retry_files)), updated_saved


def _print_config(config: RuntimeConfig) -> None:
    print("Resolved fulltext configuration:")
    for field in fields(config):
        print(f"{field.name}={getattr(config, field.name)}")


def _select_zip_names(
    config: RuntimeConfig,
    available_names: list[str],
    saved_files: set[str],
) -> tuple[list[str], set[str], list[str]]:
    in_range = filter_zip_files_range(
        available_names,
        config.since_datetime,
        config.end_datetime,
    )
    if config.zip_selection_mode == "retry_csv":
        selected = load_retry_zip_files(
            config.retry_csv_path,
            config.year,
            in_range,
        )
        updated_saved = set(saved_files)
        updated_saved.difference_update(selected)
        return selected, updated_saved, in_range

    if config.zip_selection_mode == "incremental":
        _, watermark_datetime = load_incremental_watermark(config.state_file)
        selected, updated_saved = select_incremental_candidates(
            in_range,
            config.data_dir,
            saved_files,
            config.since_datetime,
            config.end_datetime,
            watermark_datetime,
        )
        return selected, updated_saved, in_range

    selected, updated_saved = build_zip_worklist(
        in_range,
        config.data_dir,
        saved_files,
    )
    return selected, updated_saved, in_range


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = resolve_runtime_config(args)
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return 2

    setup_main_logging(config)
    if args.config_check:
        _print_config(config)
        return 0

    try:
        available_paths = collect_zip_files(config.data_dir, config.year)
    except (FileNotFoundError, OSError) as exc:
        logging.error("ZIP discovery failed: %s", exc)
        return 1

    saved_files = load_saved_file_set(config.saved_files_path)
    try:
        selected, saved_files, in_range = _select_zip_names(
            config,
            list(available_paths),
            saved_files,
        )
    except (FileNotFoundError, ValueError) as exc:
        logging.error("ZIP selection failed: %s", exc)
        return 1

    write_saved_file_set(config.saved_files_path, saved_files)
    selected_paths = {name: available_paths[name] for name in selected}
    logging.info(
        "Selected %s ZIP file(s) in inclusive range %s to %s",
        len(selected_paths),
        config.since_datetime,
        config.end_datetime or "unbounded",
    )

    result_map = {
        name: "completed"
        for name in in_range
        if name in saved_files
    }
    last_watermark = ""
    progress = tqdm(
        total=len(selected_paths),
        desc="GDELT ZIP",
        unit="zip",
        disable=config.no_progress,
        dynamic_ncols=True,
    )
    try:
        for finished_zip, empty_total, total_count, status in run_zip_pipeline(
            selected_paths,
            config.data_dir,
            config,
        ):
            result_map[finished_zip] = status
            if status == "completed":
                saved_files.add(finished_zip)
            else:
                saved_files.discard(finished_zip)
            write_saved_file_set(config.saved_files_path, saved_files)

            if config.zip_selection_mode == "incremental":
                watermark = compute_incremental_watermark(in_range, result_map)
                if watermark and watermark != last_watermark:
                    write_incremental_watermark(config.state_file, watermark)
                    last_watermark = watermark

            logging.info(
                "ZIP %s: status=%s missing=%s total=%s",
                finished_zip,
                status,
                empty_total,
                total_count,
            )
            progress.set_postfix(status=status, missing=empty_total)
            progress.update(1)
    except Exception:
        logging.exception("Fulltext worker execution failed")
        return 1
    finally:
        progress.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
