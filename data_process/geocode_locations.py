#!/usr/bin/env python
"""Step 5: geocode every usable location query without filtering articles.

Input:
    CSV/CSV.GZ articles with article_id, strict-JSON event_time objects and
    locations arrays from spatiotemporal_extraction. Other source cells,
    article identities and event-time values are preserved unchanged.
Output:
    geocoded_articles.csv for country normalization, unresolved_geocodes.csv
    for error/pending queries, and run_summary.json. Empty-location articles
    remain in the output; unavailable coordinates are null.
Execution:
    Cache-only is the offline default. Nominatim requires explicit selection;
    requests are sequential, rate-limited and retried, with query caches and a
    disk-backed index shared across streamed chunks. Geocoding is independent
    of temporal precision, location role and historical grid suitability.

Offline injection: request_fn(query, *, endpoint, user_agent, timeout) returns
a decoded Nominatim JSON array. sleep_fn and clock_fn can also be injected.
Intermediate results live in each location object, using geocode_status (ok,
not_found, error, pending, not_requested), geocode_provider, lat, lon,
display_name, geocode_type, geocode_confidence, and geocode_success_flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from data_process.utils.process_config import StepArgumentParser, guard_step_selection

import copy
import csv
import gzip
import hashlib
import io
import json
import math
import os
import sqlite3
import tempfile
import time
import urllib.parse
import urllib.request
from collections import Counter
from contextlib import contextmanager, ExitStack

from tqdm import tqdm


DEFAULT_ENDPOINT = "https://nominatim.openstreetmap.org/search"
DEFAULT_USER_AGENT = "GeoREL/1.0 research geocoder"
OUTPUT_NAMES = ("geocoded_articles.csv", "unresolved_geocodes.csv", "run_summary.json")
QUERY_FIELDS = ("geocode_query", "normalized_name", "location_text", "admin2", "admin1", "country")
FAILURE_FIELDS = ("geocode_query", "geocode_status", "geocode_provider", "error")
CACHE_VERSION = 1
DEFAULT_CHUNK_SIZE = 128


def _reject_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite JSON number: {value}")
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(value, context):
    try:
        return json.loads(value, parse_constant=_reject_constant,
                          parse_float=_finite_float, object_pairs_hook=_unique_object)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError(f"{context}: invalid strict JSON: {exc}") from exc


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _text(value):
    return " ".join((value or "").split())


def _progress_field(value):
    # Escape console controls/non-ASCII for Windows, and bound each log field.
    escaped = json.dumps(str(value)[:160], ensure_ascii=True)[1:-1]
    return escaped[:157] + "..." if len(escaped) > 160 or len(str(value)) > 160 else escaped


def _progress(progress_fn, message):
    if progress_fn is not None:
        progress_fn(message)


def _status(status_fn, phase, **fields):
    if status_fn is not None:
        status_fn({"phase": phase, **fields})


class _TqdmStatus:
    """Render structured status on the CLI's single, externally owned bar."""

    def __init__(self, bar):
        self.bar = bar
        self.network = False
        self.cached = self.deferred = self.retries = self.errors = 0

    def __call__(self, event):
        phase = event["phase"]
        if phase == "planned":
            self.network = event["network_queries"] > 0
            self.cached, self.deferred = event["cached"], event["deferred"]
            self.bar.set_description_str("Geocoding" if self.network else "Cache/offline", refresh=False)
            self.bar.reset(total=event["network_queries"] if self.network else event["unique_queries"])
        elif phase == "retry":
            self.retries += 1
        elif phase not in {"request_start", "query", "complete"}:
            return
        counts = event.get("counts", {})
        if "error_queries" in counts:
            self.errors = counts["error_queries"]
        self.bar.set_postfix(cached=self.cached, error=self.errors,
                             deferred=self.deferred, retries=self.retries, refresh=False)
        if phase in {"query", "complete"}:
            completed = counts["new_queries" if self.network else "unique_queries"]
            if completed > self.bar.n:
                self.bar.update(completed - self.bar.n)
        if phase in {"planned", "retry", "request_start", "complete"}:
            self.bar.refresh()


def build_geocode_query(location):
    """Use an explicit query or name/admin/country; never inspect model flags."""
    explicit = _text(location.get("geocode_query"))
    if explicit:
        return explicit
    name = _text(location.get("normalized_name")) or _text(location.get("location_text"))
    parts = [name, *(_text(location.get(key)) for key in ("admin2", "admin1", "country"))]
    seen = set()
    unique = []
    for part in parts:
        if part and part.casefold() not in seen:
            seen.add(part.casefold())
            unique.append(part)
    return ", ".join(unique)


def _validate_record(record, row_number):
    article_id = record.get("article_id")
    context = f"row {row_number}, article_id={article_id!r}"
    if not isinstance(article_id, str) or not article_id.strip():
        raise ValueError(f"{context}: article_id must be a nonempty string")
    event_time = _strict_json(record.get("event_time"), context + " event_time")
    if not isinstance(event_time, dict):
        raise ValueError(f"{context}: event_time must be a JSON object")
    locations = _strict_json(record.get("locations"), context + " locations")
    if not isinstance(locations, list):
        raise ValueError(f"{context}: locations must be a JSON array")
    for index, location in enumerate(locations):
        if not isinstance(location, dict):
            raise ValueError(f"{context}: locations[{index}] must be an object")
        for field in QUERY_FIELDS:
            if location.get(field) is not None and not isinstance(location[field], str):
                raise ValueError(f"{context}: locations[{index}].{field} must be a string or null")
    return locations


def _validate_records(records):
    parsed = []
    seen = set()
    for row_number, record in enumerate(records, 2):
        locations = _validate_record(record, row_number)
        article_id = record["article_id"]
        if article_id in seen:
            raise ValueError(f"row {row_number}, article_id={article_id!r}: duplicate article_id")
        seen.add(article_id)
        parsed.append(locations)
    return parsed


def _validate_options(provider, endpoint, user_agent, attempts, retry_delay,
                      timeout, rate_limit_seconds, max_new_queries):
    if provider not in {"cache-only", "nominatim"}:
        raise ValueError("provider must be cache-only or nominatim")
    url = urllib.parse.urlsplit(endpoint)
    if (url.scheme not in {"http", "https"} or not url.hostname or url.fragment
            or url.username or url.password):
        raise ValueError("endpoint must be an HTTP(S) URL without credentials or fragment")
    if not isinstance(user_agent, str) or not user_agent.strip() or any(c in user_agent for c in "\r\n"):
        raise ValueError("user-agent must be a nonempty single line")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 100:
        raise ValueError("attempts must be an integer between 1 and 100")
    for name, value in (("retry-delay", retry_delay), ("timeout", timeout),
                        ("rate-limit-seconds", rate_limit_seconds)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if timeout == 0:
        raise ValueError("timeout must be greater than zero")
    if max_new_queries is not None and (isinstance(max_new_queries, bool)
            or not isinstance(max_new_queries, int) or max_new_queries < 0):
        raise ValueError("max-new-queries must be a non-negative integer")


@contextmanager
def _atomic_text(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_write(path, write_fn):
    with _atomic_text(path) as stream:
        write_fn(stream)


def _write_json(path, value):
    _atomic_write(path, lambda stream: json.dump(value, stream, ensure_ascii=False,
                                               indent=2, allow_nan=False))


def _write_csv(path, records, fields):
    def write(stream):
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    _atomic_write(path, write)


def _query_identity(query, endpoint):
    # Cache-only consumes the same backend namespace, not a separate cache.
    return {"provider": "nominatim", "endpoint": endpoint, "query": query}


def _result(status, *, error=""):
    result = {"geocode_status": status, "geocode_provider": "nominatim",
              "lat": None, "lon": None, "display_name": "", "geocode_type": "",
              "geocode_confidence": "", "geocode_success_flag": False}
    if error:
        result["geocode_error"] = error
    return result


def _read_cached(cache, query, endpoint):
    if cache is None:
        return None
    identity = _query_identity(query, endpoint)
    path = Path(cache) / (_fingerprint(identity) + ".json")
    if not path.exists():
        return None
    try:
        payload = _strict_json(path.read_text(encoding="utf-8"), f"cache {path}")
        if (not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION
                or payload.get("identity") != identity):
            raise ValueError("cache identity or version mismatch")
        result = payload.get("result")
        required = set(_result("ok"))
        if (not isinstance(result, dict) or not required.issubset(result)
                or result["geocode_status"] not in {"ok", "not_found", "error"}
                or result["geocode_provider"] != "nominatim"):
            raise ValueError("invalid cached result schema")
        if result["geocode_status"] == "ok":
            if any(type(result[field]) not in (int, float) for field in ("lat", "lon")):
                raise ValueError("cached coordinates must be JSON numbers")
            _coordinates(result["lat"], result["lon"])
            if result["geocode_success_flag"] is not True:
                raise ValueError("successful cache result must have a true success flag")
        elif (result["lat"] is not None or result["lon"] is not None
              or result["geocode_success_flag"] is not False):
            raise ValueError("negative cache result must have null coordinates and false success flag")
        for field in ("display_name", "geocode_type", "geocode_confidence"):
            if not isinstance(result[field], str):
                raise ValueError(f"invalid cache field {field}")
        return result
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid cache {path}: {exc}") from exc


def _cache_result(cache, query, endpoint, result):
    if cache is not None:
        identity = _query_identity(query, endpoint)
        _write_json(Path(cache) / (_fingerprint(identity) + ".json"),
                    {"version": CACHE_VERSION, "identity": identity, "result": result})


def _coordinates(lat, lon):
    if isinstance(lat, bool) or isinstance(lon, bool):
        raise ValueError("provider coordinates cannot be booleans")
    lat, lon = float(lat), float(lon)
    if not math.isfinite(lat) or not math.isfinite(lon) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("provider coordinates are non-finite or out of range")
    return lat, lon


def _parse_response(payload):
    if not isinstance(payload, list):
        raise ValueError("provider response must be a JSON array")
    if not payload:
        return _result("not_found")
    item = payload[0]
    if not isinstance(item, dict):
        raise ValueError("provider response item must be an object")
    lat, lon = _coordinates(item.get("lat"), item.get("lon"))
    for field in ("display_name", "type"):
        if item.get(field) is not None and not isinstance(item[field], str):
            raise ValueError(f"provider {field} must be a string")
    return {**_result("ok"), "lat": lat, "lon": lon,
            "display_name": item.get("display_name") or "",
            "geocode_type": item.get("type") or "",
            "geocode_confidence": "nominatim_top1", "geocode_success_flag": True}


def _request_json(query, *, endpoint, user_agent, timeout):
    url = urllib.parse.urlsplit(endpoint)
    params = dict(urllib.parse.parse_qsl(url.query, keep_blank_values=True))
    params.update(q=query, format="jsonv2", limit="1", addressdetails="1")
    request_url = urllib.parse.urlunsplit(url._replace(query=urllib.parse.urlencode(params)))
    request = urllib.request.Request(request_url, headers={"User-Agent": user_agent,
                                                          "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        # Bound memory use even for a misconfigured endpoint.
        body = response.read(2_000_001)
    if len(body) > 2_000_000:
        raise ValueError("provider response exceeds 2 MB")
    return _strict_json(body.decode("utf-8"), "provider response")


class _RequestStream:
    """One request clock and query budget shared by all chunks in a run."""

    def __init__(self, *, provider, cache, endpoint, user_agent, attempts,
                 retry_delay, timeout, rate_limit_seconds, max_new_queries,
                 dry_run, request_fn, sleep_fn, clock_fn, progress_fn=None, status_fn=None):
        self.provider, self.cache, self.endpoint = provider, cache, endpoint
        self.user_agent, self.attempts = user_agent, attempts
        self.retry_delay, self.timeout = retry_delay, timeout
        self.interval = max(1.0, rate_limit_seconds)
        self.max_new_queries, self.dry_run = max_new_queries, dry_run
        self.request_fn = request_fn or _request_json
        self.sleep_fn = sleep_fn or time.sleep
        self.clock_fn = clock_fn or time.monotonic
        self.progress_fn = progress_fn
        self.status_fn = status_fn
        self.budget_reported = False
        self.last_progress_unique = None
        self.last_request = None
        self.counts = Counter(articles=0, locations=0, unique_queries=0, cache_hits=0,
                              new_queries=0, requests=0, unqueried_locations=0,
                              ok_queries=0, not_found_queries=0, error_queries=0, pending_queries=0)

    def report_progress(self, *, final=False):
        unique = self.counts["unique_queries"]
        if (self.progress_fn is None or unique == self.last_progress_unique
                or (not final and unique % 100)):
            return
        self.last_progress_unique = unique
        counts = self.counts
        _progress(self.progress_fn,
                  f"Queries: unique={unique} cache_hits={counts['cache_hits']} "
                  f"new={counts['new_queries']} requests={counts['requests']} "
                  f"ok={counts['ok_queries']} not_found={counts['not_found_queries']} "
                  f"error={counts['error_queries']} pending={counts['pending_queries']}")

    def lookup(self, query, cached):
        self.counts["unique_queries"] += 1
        if cached is not None and cached["geocode_status"] in {"ok", "not_found"}:
            self.counts["cache_hits"] += 1
            result = cached
        elif (self.provider == "cache-only" or self.dry_run
              or (self.max_new_queries is not None and self.counts["new_queries"] >= self.max_new_queries)):
            if (self.provider != "cache-only" and not self.dry_run and not self.budget_reported):
                self.budget_reported = True
                _progress(self.progress_fn,
                          f"New-query budget exhausted: limit={self.max_new_queries}; "
                          "remaining uncached queries stay pending")
            result = _result("pending", error="Network disabled or new-query budget exhausted")
        else:
            self.counts["new_queries"] += 1
            for attempt in range(self.attempts):
                if attempt:
                    _progress(self.progress_fn,
                              f"Retry: query=\"{_progress_field(query)}\" "
                              f"attempt={attempt + 1}/{self.attempts} "
                              f"backoff={self.retry_delay * (2 ** (attempt - 1))}s")
                    _status(self.status_fn, "retry", attempt=attempt + 1,
                            attempts=self.attempts, counts=dict(self.counts))
                    self.sleep_fn(self.retry_delay * (2 ** (attempt - 1)))
                if self.last_request is not None:
                    delay = self.interval - (self.clock_fn() - self.last_request)
                    if delay > 0:
                        self.sleep_fn(delay)
                _progress(self.progress_fn,
                          f"Request start: query=\"{_progress_field(query)}\" "
                          f"attempt={attempt + 1}/{self.attempts} timeout={self.timeout}s")
                _status(self.status_fn, "request_start", attempt=attempt + 1,
                        attempts=self.attempts, counts=dict(self.counts))
                self.counts["requests"] += 1
                self.last_request = self.clock_fn()
                try:
                    payload = self.request_fn(query, endpoint=self.endpoint,
                                              user_agent=self.user_agent, timeout=self.timeout)
                    result = _parse_response(payload)
                except Exception as exc:
                    result = _result("error", error=f"{type(exc).__name__}: {exc}")
                    # Exception text can contain endpoint credentials or response bodies.
                    _progress(self.progress_fn,
                              f"Request failed: query=\"{_progress_field(query)}\" "
                              f"attempt={attempt + 1}/{self.attempts} "
                              f"error={_progress_field(type(exc).__name__)}")
                _cache_result(self.cache, query, self.endpoint, result)
                if result["geocode_status"] != "error":
                    _progress(self.progress_fn,
                              f"Request result: query=\"{_progress_field(query)}\" "
                              f"status={result['geocode_status']}")
                    break
        self.counts[result["geocode_status"] + "_queries"] += 1
        self.report_progress()
        _status(self.status_fn, "query", counts=dict(self.counts))
        return result


def geocode_records(records, *, provider="cache-only", cache=None,
                    endpoint=DEFAULT_ENDPOINT, user_agent=DEFAULT_USER_AGENT,
                    attempts=3, retry_delay=1.0, timeout=30.0,
                    rate_limit_seconds=1.0, max_new_queries=None, dry_run=False,
                    request_fn=None, sleep_fn=None, clock_fn=None):
    """Return (enriched_records, unique_query_failures, execution_counts).

    No input objects are mutated. Validate the entire batch and its relevant
    cache entries before writing cache files or making a request. Successful
    and not_found cache entries are reused; errors are retried on every run.
    A budget counts unique queries, not individual attempts.
    """
    _validate_options(provider, endpoint, user_agent, attempts, retry_delay,
                      timeout, rate_limit_seconds, max_new_queries)
    records = list(records)
    parsed = _validate_records(records)
    queries = list(dict.fromkeys(query for locations in parsed for location in locations
                                if (query := build_geocode_query(location))))
    cached = {query: _read_cached(cache, query, endpoint) for query in queries}
    stream = _RequestStream(provider=provider, cache=cache, endpoint=endpoint, user_agent=user_agent,
                            attempts=attempts, retry_delay=retry_delay, timeout=timeout,
                            rate_limit_seconds=rate_limit_seconds, max_new_queries=max_new_queries,
                            dry_run=dry_run, request_fn=request_fn, sleep_fn=sleep_fn, clock_fn=clock_fn)
    results = {query: stream.lookup(query, cached[query]) for query in queries}
    counts = dict(stream.counts)
    counts.update(articles=len(records), locations=sum(map(len, parsed)))

    enriched = []
    for record, locations in zip(records, parsed):
        row = copy.deepcopy(record)
        for location in locations:
            query = build_geocode_query(location)
            location.setdefault("geocode_query", query)
            location.pop("geocode_error", None)
            location.update(results[query] if query else _result("not_requested"))
        row["locations"] = json.dumps(locations, ensure_ascii=False, allow_nan=False)
        enriched.append(row)
    failures = [{"geocode_query": query, "geocode_status": result["geocode_status"],
                 "geocode_provider": result["geocode_provider"],
                 "error": result.get("geocode_error", "")}
                for query, result in results.items() if result["geocode_status"] in {"error", "pending"}]
    statuses = Counter(result["geocode_status"] for result in results.values())
    for status in ("ok", "not_found", "error", "pending"):
        counts[status + "_queries"] = statuses[status]
    counts["unqueried_locations"] = sum(not build_geocode_query(loc) for locs in parsed for loc in locs)
    counts["exit_code"] = 2 if failures else 0
    return enriched, failures, counts


def _within(path, parent):
    return path == parent or parent in path.parents


def _input_files(input_path):
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(path.resolve() for path in input_path.iterdir()
                       if path.is_file() and path.name.lower().endswith((".csv", ".csv.gz")))
    else:
        raise ValueError(f"Input does not exist: {input_path}")
    if not files:
        raise ValueError(f"No CSV or CSV.GZ inputs found: {input_path}")
    for path in files:
        if not path.name.lower().endswith((".csv", ".csv.gz")):
            raise ValueError(f"Input must be CSV or CSV.GZ: {path}")
        if path.name.lower().removesuffix(".gz") in OUTPUT_NAMES:
            raise ValueError(f"Reserved geocoder output cannot be used as input: {path}")
    return files


def _check_paths(input_path, files, output_dir, cache):
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("output-dir must be a directory separate from input")
    if (_within(input_path, output_dir)
            or (input_path.is_dir() and _within(output_dir, input_path))
            or any(_within(path, output_dir) for path in files)):
        raise ValueError("Input and output paths must not overlap")
    for path in files:
        if (cache == path or _within(path, cache)
                or (input_path.is_dir() and _within(cache, input_path))):
            raise ValueError("Cache and input paths must not overlap")
    if cache.exists() and not cache.is_dir():
        raise ValueError("cache must be a directory")
    if _within(output_dir, cache):
        raise ValueError("Cache must not contain the output directory")
    for name in OUTPUT_NAMES:
        target = output_dir / name
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError(f"Reserved output path is not a regular file: {target}")
        if _within(cache, target):
            raise ValueError(f"Cache conflicts with reserved output: {target}")


class _HashingReader(io.RawIOBase):
    """Hash the raw bytes as CSV/gzip consumes them through bounded buffers."""

    def __init__(self, source, digest):
        self.source, self.digest = source, digest

    def readable(self):
        return True

    def readinto(self, buffer):
        count = self.source.readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count])
        return count


@contextmanager
def _csv_reader(path, digest):
    # Full news text and nested JSON routinely exceed csv's 128 KiB default.
    csv.field_size_limit(max(csv.field_size_limit(), 2_147_483_647))
    try:
        with ExitStack() as stack:
            raw = stack.enter_context(path.open("rb", buffering=0))
            buffered = stack.enter_context(io.BufferedReader(_HashingReader(raw, digest)))
            binary = stack.enter_context(gzip.GzipFile(fileobj=buffered)) if path.name.lower().endswith(".gz") else buffered
            text = stack.enter_context(io.TextIOWrapper(binary, encoding="utf-8-sig", newline=""))
            reader = csv.DictReader(text, strict=True)
            header = reader.fieldnames
            if not header or len(header) != len(set(header)) or "" in header:
                raise ValueError("missing, empty, or duplicate CSV header")
            if not {"article_id", "event_time", "locations"}.issubset(header):
                raise ValueError("CSV requires article_id, event_time, locations columns")
            yield reader
    except (OSError, EOFError, UnicodeError, csv.Error, ValueError) as exc:
        raise ValueError(f"Invalid input {path}: {exc}") from exc


def _checked_row(row, line_number):
    if None in row or any(value is None for value in row.values()):
        raise ValueError(f"row {line_number}: CSV column count mismatch")
    return _validate_record(row, line_number)


class _RunIndex:
    """Ephemeral disk index: no full article text or unbounded Python sets."""

    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA cache_size=-2048")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute("PRAGMA journal_mode=OFF")
        self.db.execute("CREATE TABLE article_ids (article_id TEXT PRIMARY KEY) WITHOUT ROWID")
        self.db.execute("CREATE TABLE queries (query TEXT PRIMARY KEY, cached TEXT, result TEXT) WITHOUT ROWID")

    def add_article(self, article_id, row_number):
        try:
            self.db.execute("INSERT INTO article_ids VALUES (?)", (article_id,))
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"row {row_number}, article_id={article_id!r}: duplicate article_id") from exc

    def add_query(self, query, cache, endpoint):
        cursor = self.db.execute("INSERT OR IGNORE INTO queries (query) VALUES (?)", (query,))
        if cursor.rowcount:
            cached = _read_cached(cache, query, endpoint)
            if cached is not None:
                self.db.execute("UPDATE queries SET cached=? WHERE query=?", (_canonical(cached), query))

    def import_legacy_cache(self, cache, endpoint, roots):
        """Promote requested terminal results without modifying legacy caches."""
        if cache is None:
            return
        caches = [Path(root) / "cache" for root in roots]
        for query, cached in self.db.execute("SELECT query, cached FROM queries ORDER BY query"):
            if cached is not None and json.loads(cached)["geocode_status"] in {"ok", "not_found"}:
                continue
            for legacy in caches:
                path = legacy / (_fingerprint(_query_identity(query, endpoint)) + ".json")
                if legacy.is_symlink() or path.is_symlink() or legacy.resolve() == Path(cache).resolve():
                    continue
                try:
                    result = _read_cached(legacy, query, endpoint)
                except (OSError, ValueError):
                    continue
                if result is None or result["geocode_status"] not in {"ok", "not_found"}:
                    continue
                current = _read_cached(cache, query, endpoint)
                if current is not None and current["geocode_status"] in {"ok", "not_found"}:
                    result = current
                else:
                    _cache_result(cache, query, endpoint, result)
                self.db.execute("UPDATE queries SET cached=? WHERE query=?", (_canonical(result), query))
                break

    def resolve(self, query, stream):
        row = self.db.execute("SELECT cached, result FROM queries WHERE query=?", (query,)).fetchone()
        if row is None:
            raise ValueError("Input query changed after preflight; retry with stable inputs")
        cached, result = row
        if result is not None:
            return json.loads(result), False
        result = stream.lookup(query, json.loads(cached) if cached is not None else None)
        self.db.execute("UPDATE queries SET result=? WHERE query=?", (_canonical(result), query))
        return result, True


def _load_inputs(files, index, cache, endpoint, progress_fn=None):
    """Validate and hash every input before requests; retain metadata only."""
    fields, identities = [], []
    articles = locations_count = 0
    for file_number, path in enumerate(files, 1):
        file_articles = 0
        report_file = file_number in (1, len(files)) or file_number % 10 == 0
        if report_file:
            _progress(progress_fn, f"Preflight: file={file_number}/{len(files)} start")
        digest = hashlib.sha256()
        with _csv_reader(path, digest) as reader:
            fields.extend(field for field in reader.fieldnames if field not in fields)
            for row in reader:
                locations = _checked_row(row, reader.line_num)
                index.add_article(row["article_id"], reader.line_num)
                articles += 1
                file_articles += 1
                locations_count += len(locations)
                for location in locations:
                    query = build_geocode_query(location)
                    if query:
                        index.add_query(query, cache, endpoint)
                if file_articles % 1000 == 0:
                    _progress(progress_fn,
                              f"Preflight: file={file_number}/{len(files)} "
                              f"file_articles={file_articles} articles={articles} locations={locations_count}")
        identities.append({"path": str(path), "sha256": digest.hexdigest()})
        if report_file:
            _progress(progress_fn,
                      f"Preflight: file={file_number}/{len(files)} complete "
                      f"file_articles={file_articles} articles={articles} locations={locations_count}")
    index.db.commit()
    if progress_fn is not None:
        unique = index.db.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
        _progress(progress_fn,
                  f"Inputs validated: files={len(files)} articles={articles} "
                  f"locations={locations_count} unique={unique}")
    return fields, _fingerprint(identities)


def _input_chunks(files, chunk_size, expected_fingerprint):
    chunk, identities = [], []
    for path in files:
        digest = hashlib.sha256()
        with _csv_reader(path, digest) as reader:
            for row in reader:
                chunk.append((row, _checked_row(row, reader.line_num)))
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
        identities.append({"path": str(path), "sha256": digest.hexdigest()})
    if chunk:
        yield chunk
    if _fingerprint(identities) != expected_fingerprint:
        raise ValueError("Input changed after preflight; outputs were not replaced; retry with stable inputs")


def _write_streaming_outputs(files, fields, output_dir, input_fingerprint, chunk_size, index, stream,
                             progress_fn=None):
    chunks = 0
    last_progress_chunk = None
    with _atomic_text(output_dir / "geocoded_articles.csv") as articles, \
            _atomic_text(output_dir / "unresolved_geocodes.csv") as unresolved:
        article_writer = csv.DictWriter(articles, fieldnames=fields)
        failure_writer = csv.DictWriter(unresolved, fieldnames=FAILURE_FIELDS)
        article_writer.writeheader()
        failure_writer.writeheader()
        for chunk in _input_chunks(files, chunk_size, input_fingerprint):
            chunks += 1
            for row, locations in chunk:
                stream.counts["articles"] += 1
                stream.counts["locations"] += len(locations)
                for location in locations:
                    query = build_geocode_query(location)
                    if query:
                        result, first = index.resolve(query, stream)
                        if first and result["geocode_status"] in {"error", "pending"}:
                            failure_writer.writerow({"geocode_query": query,
                                                     "geocode_status": result["geocode_status"],
                                                     "geocode_provider": result["geocode_provider"],
                                                     "error": result.get("geocode_error", "")})
                    else:
                        result = _result("not_requested")
                        stream.counts["unqueried_locations"] += 1
                    location.setdefault("geocode_query", query)
                    location.pop("geocode_error", None)
                    location.update(result)
                row["locations"] = json.dumps(locations, ensure_ascii=False, allow_nan=False)
                article_writer.writerow(row)
            index.db.commit()
            if chunks == 1 or chunks % 10 == 0:
                _progress(progress_fn,
                          f"Articles: chunks={chunks} articles={stream.counts['articles']} "
                          f"locations={stream.counts['locations']}")
                last_progress_chunk = chunks
    if chunks != last_progress_chunk:
        _progress(progress_fn,
                  f"Articles: chunks={chunks} articles={stream.counts['articles']} "
                  f"locations={stream.counts['locations']}")
    stream.report_progress(final=True)
    counts = dict(stream.counts)
    counts["exit_code"] = 2 if counts["error_queries"] or counts["pending_queries"] else 0
    return counts


def run_geocoding(input_path, output_dir, *, provider="cache-only", cache=None,
                  endpoint=DEFAULT_ENDPOINT, user_agent=DEFAULT_USER_AGENT,
                  attempts=3, retry_delay=1.0, timeout=30.0,
                  rate_limit_seconds=1.0, max_new_queries=None, dry_run=False,
                  request_fn=None, sleep_fn=None, clock_fn=None, chunk_size=DEFAULT_CHUNK_SIZE,
                  legacy_cache_roots=(), progress_fn=None, status_fn=None):
    """Stream preflight and bounded chunks with disk-backed query deduplication.

    The run_summary is also the ownership/resume marker, written atomically
    before requests so an interrupted run can reuse its partial query cache.
    Changed input paths, provider, endpoint, user-agent, or cache path require a
    new output directory. Revised bytes at the same paths rebuild the current
    aggregates, retaining the previous summary under input_revisions. Dry-run,
    retry, timeout, rate, budget, and chunk-size changes do not change run
    identity. Results are streamed to temporary CSVs
    and only published after the complete processing pass succeeds.
    progress_fn, when supplied, receives bounded ASCII diagnostic lines; the
    default is quiet. status_fn optionally receives dictionaries with phase
    (preparing, planned, request_start, retry, query, complete) and independent
    counts snapshots during execution. The planned event reports unique_queries,
    network_queries, cached and deferred after legacy-cache import. Query events
    follow final retries, not individual HTTP attempts. Neither callback affects
    run/cache identity; callback exceptions propagate without provider retries.
    """
    _progress(progress_fn,
              f"Geocoding start: provider={_progress_field(provider)} dry_run={dry_run}")
    _status(status_fn, "preparing")
    _validate_options(provider, endpoint, user_agent, attempts, retry_delay,
                      timeout, rate_limit_seconds, max_new_queries)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk-size must be a positive integer")
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    cache = Path(cache).expanduser().resolve() if cache is not None else output_dir / "cache"
    legacy_cache_roots = tuple(dict.fromkeys(Path(root).expanduser().absolute() for root in legacy_cache_roots))
    files = _input_files(input_path)
    _check_paths(input_path, files, output_dir, cache)
    config = {"provider": provider, "cache": str(cache), "endpoint": endpoint,
              "user_agent": user_agent, "attempts": attempts, "retry_delay": retry_delay,
              "timeout": timeout, "rate_limit_seconds": max(1.0, rate_limit_seconds),
              "max_new_queries": max_new_queries, "dry_run": dry_run, "chunk_size": chunk_size}
    identity_config = {key: config[key] for key in ("provider", "cache", "endpoint", "user_agent")}
    with tempfile.TemporaryDirectory(prefix="georel-geocode-") as temporary:
        index = _RunIndex(Path(temporary) / "run.sqlite3")
        try:
            fields, input_fingerprint = _load_inputs(files, index, cache, endpoint, progress_fn)
            run_fingerprint = _fingerprint({"version": CACHE_VERSION, "input": input_fingerprint,
                                            "config": identity_config})
            summary_path = output_dir / "run_summary.json"
            if summary_path.exists():
                previous = _strict_json(summary_path.read_text(encoding="utf-8"), "run summary")
                old_config = previous.get("configuration", {}) if isinstance(previous, dict) else {}
                # Older summaries without paths may only resume identical bytes.
                same_inputs = (isinstance(previous, dict) and (
                    previous.get("input_files") == [str(path) for path in files]
                    or ("input_files" not in previous
                        and previous.get("input_fingerprint") == input_fingerprint)))
                compatible = (isinstance(previous, dict) and isinstance(old_config, dict)
                              and same_inputs
                              and {key: old_config.get(key) for key in identity_config} == identity_config)
                if (not compatible and isinstance(previous, dict) and isinstance(old_config, dict)
                        and output_dir in legacy_cache_roots
                        and same_inputs
                        and old_config.get("cache") == str(output_dir / "cache")
                        and all(old_config.get(key) == identity_config[key] for key in ("provider", "endpoint", "user_agent"))):
                    # Only the historical per-output cache may relocate in place.
                    compatible = True
                if not compatible:
                    raise ValueError("Output conflicts with different input/configuration; use a new output directory")
                if previous.get("input_fingerprint") != input_fingerprint:
                    _write_json(output_dir / "input_revisions" / _fingerprint(previous) / "run_summary.json",
                                previous)
            elif output_dir.exists() and any(path.name != "cache" or path.resolve() != cache
                                             for path in output_dir.iterdir()):
                raise ValueError("Output directory contains unowned files; use a new output directory")
            index.import_legacy_cache(cache, endpoint, legacy_cache_roots)
            if status_fn is not None:
                unique = cached = 0
                for (value,) in index.db.execute("SELECT cached FROM queries"):
                    unique += 1
                    if value is not None and json.loads(value)["geocode_status"] in {"ok", "not_found"}:
                        cached += 1
                network = unique - cached if provider == "nominatim" and not dry_run else 0
                if max_new_queries is not None:
                    network = min(network, max_new_queries)
                _status(status_fn, "planned", unique_queries=unique, network_queries=network,
                        cached=cached, deferred=unique - cached - network)
            metadata = {"schema_version": CACHE_VERSION, "input_fingerprint": input_fingerprint,
                        "run_fingerprint": run_fingerprint, "configuration": config,
                        "input_files": [str(path) for path in files]}
            _write_json(summary_path, {**metadata, "state": "running"})
            stream = _RequestStream(
                provider=provider, cache=cache, endpoint=endpoint, user_agent=user_agent,
                attempts=attempts, retry_delay=retry_delay, timeout=timeout,
                rate_limit_seconds=rate_limit_seconds, max_new_queries=max_new_queries,
                dry_run=dry_run, request_fn=request_fn, sleep_fn=sleep_fn, clock_fn=clock_fn,
                progress_fn=progress_fn, status_fn=status_fn)
            counts = _write_streaming_outputs(files, fields, output_dir, input_fingerprint,
                                              chunk_size, index, stream, progress_fn)
            summary = {**metadata, **counts, "state": "complete"}
            _write_json(summary_path, summary)
            _status(status_fn, "complete", counts=dict(counts))
        finally:
            index.db.close()
    return summary


def build_parser():
    parser = StepArgumentParser(step="5_geocode_locations", description=__doc__)
    parser.add_argument("--input", required=True, help="CSV, CSV.GZ, or directory of CSV files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--provider", choices=("cache-only", "nominatim"), default="cache-only",
                        help="Changing provider requires a new output directory")
    parser.add_argument("--cache", help="Cache directory; default: OUTPUT_DIR/cache")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--rate-limit-seconds", type=float, default=1.0)
    parser.add_argument("--max-new-queries", "--max-new-geocodes", dest="max_new_queries", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="Maximum article rows buffered at once (default: 128)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        with tqdm(total=None, desc="Preparing", unit="query", ascii=True,
                  dynamic_ncols=True, disable=False) as bar:
            guard_step_selection(args)
            summary = run_geocoding(args.input, args.output_dir, provider=args.provider,
                                    cache=args.cache, endpoint=args.endpoint, user_agent=args.user_agent,
                                    attempts=args.attempts, retry_delay=args.retry_delay,
                                    timeout=args.timeout, rate_limit_seconds=args.rate_limit_seconds,
                                    max_new_queries=args.max_new_queries, dry_run=args.dry_run,
                                    chunk_size=args.chunk_size, status_fn=_TqdmStatus(bar),
                                    legacy_cache_roots=(getattr(args, "legacy_cache_roots", ())
                                                        if getattr(args, "shared_cache_dir", None) is not None else ()))
    except (OSError, ValueError, csv.Error, sqlite3.Error) as exc:
        print(f"Geocoding error: {exc}", file=sys.stderr)
        return 2
    print(f"Articles={summary['articles']}, queries={summary['unique_queries']}, "
          f"ok={summary['ok_queries']}, not_found={summary['not_found_queries']}, "
          f"error={summary['error_queries']}, pending={summary['pending_queries']}", flush=True)
    return summary["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
