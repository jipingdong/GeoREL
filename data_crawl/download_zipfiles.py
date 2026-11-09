import argparse
import csv
import hashlib
import json
import logging
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener


GDELT_BASE_URL = "http://data.gdeltproject.org/gdeltv2"
MASTERFILELIST_URL = f"{GDELT_BASE_URL}/masterfilelist.txt"
EXPORT_SUFFIX = "export.CSV"
ZIP_DIR = "gdelt_zip"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class GdeltFile:
    timestamp: str
    url: str
    expected_size: Optional[int] = None
    md5: Optional[str] = None

    @property
    def filename(self) -> str:
        return f"{self.timestamp}.zip"


class SafeStreamHandler(logging.StreamHandler):
    def handleError(self, record):
        return

    def flush(self):
        try:
            super().flush()
        except OSError:
            return


class SafeFileHandler(logging.FileHandler):
    def handleError(self, record):
        return

    def flush(self):
        try:
            super().flush()
        except OSError:
            return


def setup_logging(data_dir: Path) -> None:
    log_dir = data_dir / ZIP_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except OSError:
            pass

    root_logger.setLevel(logging.INFO)
    logging.raiseExceptions = False

    handlers: List[logging.Handler] = [SafeStreamHandler(sys.stdout)]
    try:
        handlers.append(SafeFileHandler(log_dir / "gdelt_download.log", encoding="utf-8"))
    except OSError as exc:
        print(f"Warning: file logging disabled: {exc}", file=sys.stderr)

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        root_logger.addHandler(handler)


def parse_datetime_arg(value: Optional[str], is_end: bool = False) -> Optional[datetime]:
    if not value:
        return None

    value = value.strip()
    if len(value) == 14 and value.isdigit():
        return datetime.strptime(value, "%Y%m%d%H%M%S")

    if len(value) == 8 and value.isdigit():
        parsed = datetime.strptime(value, "%Y%m%d")
        return parsed.replace(hour=23, minute=59, second=59) if is_end else parsed

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid datetime '{value}'. Use YYYYMMDDHHMMSS, YYYYMMDD, "
            "YYYY-MM-DD, or YYYY-MM-DDTHH:MM:SS."
        ) from exc

    if is_end and len(value) == 10:
        parsed = parsed.replace(hour=23, minute=59, second=59)
    return parsed


def in_time_range(timestamp: str, start: Optional[datetime], end: Optional[datetime]) -> bool:
    dt = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
    if start and dt < start:
        return False
    if end and dt > end:
        return False
    return True


def ceil_to_15_minutes(dt: datetime) -> datetime:
    minute = (dt.minute // 15) * 15
    floored = dt.replace(minute=minute, second=0, microsecond=0)
    if floored < dt:
        floored += timedelta(minutes=15)
    return floored


def build_gdelt_url(timestamp: str) -> str:
    return f"{GDELT_BASE_URL}/{timestamp}.{EXPORT_SUFFIX}.zip"


def parse_record_url(url: str) -> Optional[str]:
    name = Path(urlparse(url).path).name
    parts = name.split(".")
    if len(parts) < 4 or not parts[0].isdigit() or len(parts[0]) != 14:
        return None

    if ".export.CSV.zip" in name:
        return parts[0]
    return None


def dataset_dir(data_dir: Path) -> Path:
    return data_dir / ZIP_DIR


def local_path(data_dir: Path, record: GdeltFile) -> Path:
    return (
        dataset_dir(data_dir)
        / record.timestamp[:4]
        / record.timestamp[4:6]
        / record.filename
    )


def has_valid_local_file(data_dir: Path, record: GdeltFile, min_valid_bytes: int) -> bool:
    path = local_path(data_dir, record)
    return path.exists() and path.stat().st_size >= min_valid_bytes


def build_opener_for(proxy: str):
    handlers = []
    if proxy:
        handlers.append(ProxyHandler({"http": proxy, "https": proxy}))
    return build_opener(*handlers)


def http_get_to_file(
    url: str,
    destination: Path,
    proxy: str,
    timeout: int,
    retries: int,
) -> None:
    opener = build_opener_for(proxy)
    request = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_suffix(destination.suffix + ".part")

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            if part_path.exists():
                part_path.unlink()

            with opener.open(request, timeout=timeout) as response, open(part_path, "wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)

            part_path.replace(destination)
            return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if part_path.exists():
                part_path.unlink()
            if attempt < retries:
                time.sleep(min(30, 2 ** attempt))

    raise RuntimeError(f"download failed after {retries} tries: {last_error}")


def fetch_masterfilelist(
    data_dir: Path,
    proxy: str,
    timeout: int,
    retries: int,
    refresh: bool,
) -> Path:
    master_path = data_dir / ZIP_DIR / "masterfilelist.txt"
    if master_path.exists() and not refresh:
        logging.info("Use local masterfilelist: %s", master_path)
        return master_path

    logging.info("Download masterfilelist: %s", MASTERFILELIST_URL)
    http_get_to_file(MASTERFILELIST_URL, master_path, proxy, timeout, retries)
    return master_path


def read_master_records(
    master_path: Path,
    start: Optional[datetime],
    end: Optional[datetime],
) -> List[GdeltFile]:
    records: List[GdeltFile] = []

    with open(master_path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            timestamp = parse_record_url(parts[2])
            if not timestamp or not in_time_range(timestamp, start, end):
                continue

            expected_size = int(parts[0]) if parts[0].isdigit() else None
            records.append(
                GdeltFile(
                    timestamp=timestamp,
                    url=parts[2],
                    expected_size=expected_size,
                    md5=parts[1],
                )
            )

    return sorted(records, key=lambda item: item.timestamp)


def generate_records(start: datetime, end: datetime) -> List[GdeltFile]:
    records: List[GdeltFile] = []
    cursor = ceil_to_15_minutes(start)
    while cursor <= end:
        timestamp = cursor.strftime("%Y%m%d%H%M%S")
        records.append(GdeltFile(timestamp, build_gdelt_url(timestamp)))
        cursor += timedelta(minutes=15)
    return records


def read_json(path: Path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)


def state_path(data_dir: Path) -> Path:
    return data_dir / ZIP_DIR / "state" / "download_state.json"


def attempts_path(data_dir: Path) -> Path:
    return data_dir / ZIP_DIR / "state" / "download_attempts.csv"


def empty_manifest_path(data_dir: Path) -> Path:
    return data_dir / ZIP_DIR / "state" / "empty_files.csv"


def failed_manifest_path(data_dir: Path) -> Path:
    return data_dir / ZIP_DIR / "state" / "failed_files.csv"


def load_state(data_dir: Path) -> Dict:
    return read_json(state_path(data_dir), {})


def save_state(data_dir: Path, state: Dict) -> None:
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(state_path(data_dir), state)


def last_success_timestamp(state: Dict) -> Optional[str]:
    value = state.get("last_success_timestamp")
    return str(value) if value else None


def has_incremental_state(state: Dict) -> bool:
    return bool(last_success_timestamp(state))


def append_attempts(data_dir: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return

    path = attempts_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_at",
        "timestamp",
        "url",
        "path",
        "status",
        "bytes",
        "expected_size",
        "error",
    ]
    should_write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_manifest(path: Path) -> List[GdeltFile]:
    if not path.exists():
        return []

    records: List[GdeltFile] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = (row.get("timestamp") or "").strip()
            url = (row.get("url") or "").strip()
            if not timestamp:
                continue
            records.append(GdeltFile(timestamp, url or build_gdelt_url(timestamp)))
    return records


def write_manifest(data_dir: Path, path: Path, records: Sequence[GdeltFile]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "url", "path"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(unique_records(records), key=lambda item: item.timestamp):
            writer.writerow(
                {
                    "timestamp": record.timestamp,
                    "url": record.url,
                    "path": str(local_path(data_dir, record)),
                }
            )


def unique_records(records: Iterable[GdeltFile]) -> List[GdeltFile]:
    by_key: Dict[str, GdeltFile] = {}
    for record in records:
        by_key[record.timestamp] = record
    return list(by_key.values())


def scan_empty_files(data_dir: Path, min_valid_bytes: int) -> List[GdeltFile]:
    records: List[GdeltFile] = []
    root = dataset_dir(data_dir)
    if not root.exists():
        return records

    for path in root.rglob("*.zip"):
        timestamp = path.stem
        if len(timestamp) != 14 or not timestamp.isdigit():
            continue
        if path.stat().st_size < min_valid_bytes:
            records.append(GdeltFile(timestamp, build_gdelt_url(timestamp)))
    return records


def unresolved_manifest_records(
    data_dir: Path,
    manifest: Path,
    min_valid_bytes: int,
) -> List[GdeltFile]:
    records = read_manifest(manifest) + scan_empty_files(data_dir, min_valid_bytes)
    unresolved: List[GdeltFile] = []
    for record in unique_records(records):
        if not has_valid_local_file(data_dir, record, min_valid_bytes):
            unresolved.append(record)
    return unresolved


def remove_successful_records(records: Sequence[GdeltFile], rows: Sequence[Dict]) -> List[GdeltFile]:
    success_keys = {
        str(row.get("timestamp"))
        for row in rows
        if row.get("status") == "success"
    }
    return [
        record
        for record in records
        if record.timestamp not in success_keys
    ]


def filter_existing(
    data_dir: Path,
    records: Sequence[GdeltFile],
    min_valid_bytes: int,
    overwrite: bool,
) -> List[GdeltFile]:
    if overwrite:
        return list(records)

    selected: List[GdeltFile] = []
    for record in records:
        if not has_valid_local_file(data_dir, record, min_valid_bytes):
            selected.append(record)
    return selected


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(
    data_dir: Path,
    record: GdeltFile,
    proxy: str,
    timeout: int,
    retries: int,
    min_valid_bytes: int,
    verify_md5: bool,
) -> Dict:
    path = local_path(data_dir, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    run_at = datetime.now().isoformat(timespec="seconds")

    row = {
        "run_at": run_at,
        "timestamp": record.timestamp,
        "url": record.url,
        "path": str(path),
        "status": "unknown",
        "bytes": 0,
        "expected_size": record.expected_size or "",
        "error": "",
    }

    try:
        http_get_to_file(record.url, path, proxy, timeout, retries)
        size = path.stat().st_size if path.exists() else 0
        row["bytes"] = size

        if size < min_valid_bytes:
            row["status"] = "empty"
        elif record.expected_size is not None and size != record.expected_size:
            row["status"] = "size_mismatch"
            row["error"] = f"expected {record.expected_size}, got {size}"
        elif verify_md5 and record.md5 and md5_file(path).lower() != record.md5.lower():
            row["status"] = "md5_mismatch"
            row["error"] = "downloaded file md5 differs from masterfilelist"
        else:
            row["status"] = "success"
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = str(exc)

    return row


def update_state_from_results(data_dir: Path, state: Dict, rows: Sequence[Dict]) -> None:
    for row in rows:
        if row.get("status") != "success":
            continue

        timestamp = str(row["timestamp"])
        current = state.get("last_success_timestamp")
        if not current or timestamp > current:
            state["last_success_timestamp"] = timestamp
        state["last_success_path"] = row.get("path", "")

    save_state(data_dir, state)


def records_from_rows(rows: Sequence[Dict], status: str) -> List[GdeltFile]:
    records: List[GdeltFile] = []
    for row in rows:
        if row.get("status") == status:
            records.append(GdeltFile(str(row["timestamp"]), str(row["url"])))
    return records


def choose_mode(args, state: Dict) -> str:
    if args.mode != "auto":
        return args.mode
    if args.start or args.end:
        return "range"
    return "incremental" if has_incremental_state(state) else "full"


def build_plan(args, data_dir: Path, state: Dict) -> Tuple[str, List[GdeltFile]]:
    start = parse_datetime_arg(args.start, is_end=False)
    end = parse_datetime_arg(args.end, is_end=True)
    if start and end and start > end:
        raise ValueError("--start must be earlier than or equal to --end")

    if args.retry_empty:
        records = unresolved_manifest_records(
            data_dir,
            empty_manifest_path(data_dir),
            args.min_valid_bytes,
        )
        return "retry-empty", records

    if args.retry_failed:
        records = read_manifest(failed_manifest_path(data_dir))
        return "retry-failed", records

    mode = choose_mode(args, state)
    if args.source == "generated":
        if not start or not end:
            raise ValueError("--source generated requires both --start and --end")
        records = generate_records(start, end)
    else:
        master_path = fetch_masterfilelist(
            data_dir,
            args.proxy,
            args.timeout,
            args.retries,
            refresh=not args.no_refresh_master,
        )
        records = read_master_records(master_path, start, end)

    if mode == "incremental":
        last_timestamp = last_success_timestamp(state)
        records = [
            record
            for record in records
            if not last_timestamp or record.timestamp > last_timestamp
        ]

    records = unique_records(records)
    records = filter_existing(data_dir, records, args.min_valid_bytes, args.overwrite)
    return mode, sorted(records, key=lambda item: item.timestamp)


def run_downloads(args, data_dir: Path, records: Sequence[GdeltFile]) -> List[Dict]:
    if args.dry_run:
        for record in records[:20]:
            print(f"{record.timestamp}\t{record.url}")
        if len(records) > 20:
            print(f"... {len(records) - 20} more")
        return []

    rows: List[Dict] = []
    total = len(records)
    if total == 0:
        return rows

    logging.info("Start downloading %s file(s) with %s worker(s)", total, args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one,
                data_dir,
                record,
                args.proxy,
                args.timeout,
                args.retries,
                args.min_valid_bytes,
                args.verify_md5,
            ): record
            for record in records
        }

        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            if row["status"] == "success":
                if index == 1 or index == total or index % args.log_every == 0:
                    logging.info("[%s/%s] success %s", index, total, row["timestamp"])
            else:
                logging.warning(
                    "[%s/%s] %s %s: %s",
                    index,
                    total,
                    row["status"],
                    row["timestamp"],
                    row.get("error", ""),
                )

    return rows


def refresh_manifests(data_dir: Path, min_valid_bytes: int, rows: Sequence[Dict]) -> None:
    empty_records = unresolved_manifest_records(
        data_dir,
        empty_manifest_path(data_dir),
        min_valid_bytes,
    )
    failed_records = read_manifest(failed_manifest_path(data_dir))

    empty_records += records_from_rows(rows, "empty")
    failed_records += records_from_rows(rows, "failed")
    failed_records += records_from_rows(rows, "size_mismatch")
    failed_records += records_from_rows(rows, "md5_mismatch")

    empty_records = remove_successful_records(empty_records, rows)
    failed_records = remove_successful_records(failed_records, rows)

    write_manifest(data_dir, empty_manifest_path(data_dir), empty_records)
    write_manifest(data_dir, failed_manifest_path(data_dir), failed_records)


def summarize(rows: Sequence[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Download GDELT v2 export.CSV zip files with full, incremental, "
            "date-range, empty-file retry, and failed-file retry modes."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data", help="Root directory for zip files and state.")
    parser.add_argument("--mode", choices=["auto", "full", "incremental", "range"], default="full", help="auto runs full on first run, incremental after state exists.")
    parser.add_argument("--source", choices=["master", "generated"], default="master", help="master reads masterfilelist.txt; generated constructs 15-minute URLs from start/end.")
    parser.add_argument("--start", default="20260801000000", help="Inclusive start time: YYYYMMDDHHMMSS, YYYYMMDD, or ISO datetime.")
    parser.add_argument("--end", default="20260801234500", help="Inclusive end time: YYYYMMDDHHMMSS, YYYYMMDD, or ISO datetime.")
    parser.add_argument("--retry-empty", action="store_true", help="Download only files recorded/scanned as 0KB or smaller than --min-valid-bytes.")
    parser.add_argument("--retry-failed", action="store_true", help="Download only files recorded as failed or invalid.")
    parser.add_argument("--overwrite", action="store_true", help="Download even when a local non-empty file already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print the selected plan without downloading.")
    parser.add_argument("--workers", type=int, default=32, help="Concurrent download worker count.")
    parser.add_argument("--timeout", type=int, default=120, help="Per-request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=5, help="Retry count per file.")
    parser.add_argument("--log-every", type=int, default=1000, help="Log one success message every N completed files. Failures are always logged.")
    parser.add_argument("--min-valid-bytes", type=int, default=1, help="Files smaller than this are treated as empty/invalid.")
    parser.add_argument("--proxy", default="", help="Optional HTTP/HTTPS proxy, e.g. http://127.0.0.1:7897.")
    parser.add_argument("--verify-md5", action="store_true", help="Verify md5 when records come from masterfilelist.txt.")
    parser.add_argument("--no-refresh-master", action="store_true", help="Use local masterfilelist.txt if present.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.data_dir = args.data_dir.resolve()
    args.workers = max(1, args.workers)
    args.retries = max(1, args.retries)
    args.log_every = max(1, args.log_every)
    args.min_valid_bytes = max(1, args.min_valid_bytes)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.data_dir)

    logging.info("Data directory: %s", args.data_dir)

    try:
        state = load_state(args.data_dir)
        mode, records = build_plan(args, args.data_dir, state)
    except Exception as exc:
        logging.error("%s", exc)
        return 2

    logging.info("Mode: %s", mode)
    logging.info("Selected %s file(s)", len(records))

    rows = run_downloads(args, args.data_dir, records)
    if args.dry_run:
        return 0

    append_attempts(args.data_dir, rows)
    update_state_from_results(args.data_dir, state, rows)
    refresh_manifests(args.data_dir, args.min_valid_bytes, rows)

    counts = summarize(rows)
    logging.info("Summary: %s", counts)
    return 0 if not any(status != "success" for status in counts) else 1


if __name__ == "__main__":
    main()
