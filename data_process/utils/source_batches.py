"""Prepare crawler month batches for relevance screening, without model calls.

Only immediate YYYYMMDDhhmmss folders containing urls_title.csv are selected.
The returned articles/ directory contains one normalized CSV per selected batch;
manifests, counts and body-free exclusion reports stay outside that directory.
Existing snapshots are reusable only when source/configuration and output hashes
match. _source_row is the CSV record number, counting the header as record 1.
Importing this module does not read inputs, create files or contact services.
"""

from contextlib import contextmanager
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile


EXPORT_COLUMN_COUNT = 61
EVENT_ID_INDEX = 0
DATEADDED_INDEX = 59
SOURCEURL_INDEX = 60
_VERSION = 1
_RESERVED = {"_source_batch", "_source_row"}
_EXCLUDED_FIELDS = ("batch", "row", "url", "reason")
_COUNTS = ("input_rows", "prepared_rows", "excluded_rows", "missing_collection_time")


def _timestamp(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9]{14}", value) is None:
        raise ValueError("Timestamps must be 14 calendar-valid digits: YYYYMMDDhhmmss.")
    try:
        datetime.strptime(value, "%Y%m%d%H%M%S")
    except ValueError as error:
        raise ValueError("Invalid calendar timestamp: " + value) from error
    return value


def _collection_time(value):
    try:
        return _timestamp(value.strip())
    except ValueError:
        return ""


def _safe_path(value):
    path = Path(value).expanduser().absolute()
    for part in (path, *path.parents):
        try:
            attributes = getattr(part.lstat(), "st_file_attributes", 0)
        except FileNotFoundError:
            attributes = 0
        if part.is_symlink() or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise ValueError("Symlinks/junctions are not permitted in preparation paths: " + str(part))
    resolved = path.resolve()
    if resolved != Path(os.path.abspath(path)):
        raise ValueError("Preparation paths must not redirect outside their directory: " + str(path))
    return resolved


def _sha(path):
    digest = hashlib.sha256()
    with _safe_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


@contextmanager
def _csv_limit():
    previous = csv.field_size_limit()
    csv.field_size_limit(max(previous, 2**31 - 1))
    try:
        yield
    finally:
        csv.field_size_limit(previous)


@contextmanager
def _atomic_text(path):
    path = _safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _safe_path(path.parent)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                     prefix=".prepare-", suffix=".tmp", dir=path.parent,
                                     delete=False) as handle:
        temporary = Path(handle.name)
        try:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        _safe_path(path)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _save(path, value):
    with _atomic_text(path) as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _load(path):
    try:
        with _safe_path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("Expected a manifest object.")
        return value
    except (OSError, ValueError) as error:
        raise ValueError("Preparation manifest integrity error: " + str(path)) from error


def _inputs(root, start, end, progress_fn=None):
    batches = []
    seen = set()
    children = sorted((path for path in root.iterdir()
                       if re.fullmatch(r"[0-9]{14}", path.name) and start <= path.name <= end),
                      key=lambda path: path.name)
    report = progress_fn or (lambda *args: None)
    for number, child in enumerate(children):
        report("Source checks", number, len(children), child.name)
        _timestamp(child.name)
        child = _safe_path(child)
        if not child.is_dir():
            continue
        if child.name in seen:
            raise ValueError("Duplicate source batch timestamp: " + child.name)
        seen.add(child.name)
        source = _safe_path(child / "urls_title.csv")
        export = _safe_path(child / (child.name + ".export.CSV"))
        if not source.is_file():
            raise ValueError("Selected batch is missing urls_title.csv: " + child.name)
        if export.exists() and not export.is_file():
            raise ValueError("Batch export must be a regular file: " + str(export))
        batches.append({"batch": child.name, "source_sha256": _sha(source),
                        "export_sha256": _sha(export) if export.is_file() else None})
        report("Source checks", number + 1, len(children), child.name)
    if not batches:
        raise ValueError("No source batch CSVs selected by the inclusive timestamp range.")
    return batches


def _header(reader, settings):
    fields = reader.fieldnames
    if not fields or any(not field or not field.strip() for field in fields) or len(fields) != len(set(fields)):
        raise ValueError("Source CSV requires unique, nonempty column headers.")
    if _RESERVED.intersection(fields):
        raise ValueError("Source CSV contains reserved _source_batch/_source_row columns.")
    if settings["status_column"] not in fields:
        raise ValueError("Source CSV is missing the extraction status column.")
    if not set(fields).intersection({"SOURCEURL", "url", "source_url"}):
        raise ValueError("Source CSV needs a SOURCEURL or url column.")
    pairs = [("TITLE", settings["title_column"]), ("RAWTEXT", settings["text_column"])]
    for canonical, alias in pairs:
        if canonical not in fields and alias not in fields:
            raise ValueError("Source CSV is missing " + canonical + " / " + alias)
    dropped = {alias for canonical, alias in pairs if alias != canonical and canonical in fields and alias in fields}
    renames = {alias: canonical for canonical, alias in pairs if canonical not in fields}
    output = [renames.get(field, field) for field in fields if field not in dropped]
    if len(output) != len(set(output)):
        raise ValueError("Source column aliases conflict with canonical headers.")
    if "collection_time" not in output:
        output.append("collection_time")
    return pairs, dropped, renames, [*output, "_source_batch", "_source_row"]


def _url(row):
    return next((row.get(name, "").strip() for name in ("SOURCEURL", "url", "source_url")
                 if row.get(name, "").strip()), "")


def _export_times(path):
    times = {}
    if not path.is_file():
        return times
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t", strict=True):
            if len(row) != EXPORT_COLUMN_COUNT:
                raise ValueError("GDELT export rows must have exactly 61 tab-separated columns.")
            timestamp, url = _collection_time(row[DATEADDED_INDEX]), row[SOURCEURL_INDEX].strip()
            if timestamp and url and (url not in times or timestamp < times[url]):
                times[url] = timestamp
    return times


def _prepare_batch(item, root, output, settings):
    stamp = item["batch"]
    source = root / stamp / "urls_title.csv"
    export = root / stamp / (stamp + ".export.CSV")
    times = _export_times(export)
    counts = dict.fromkeys(_COUNTS, 0)
    excluded = []
    target = output / "articles" / (stamp + ".csv")
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        pairs, dropped, renames, fields = _header(reader, settings)
        with _atomic_text(target) as destination:
            writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for number, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("Source CSV row width differs from its header.")
                for canonical, alias in pairs:
                    if canonical != alias and canonical in row and alias in row and row[canonical] != row[alias]:
                        raise ValueError("Source canonical/alias values conflict in batch " + stamp + ", row " + str(number))
                counts["input_rows"] += 1
                text = row["RAWTEXT"] if "RAWTEXT" in row else row[settings["text_column"]]
                status = row[settings["status_column"]].strip()
                reason = ("status_not_success:" + status if status != settings["success_status"] else
                          "empty_body" if not text.strip() else "")
                if reason:
                    excluded.append({"batch": stamp, "row": number, "url": _url(row), "reason": reason})
                    counts["excluded_rows"] += 1
                    continue
                normalized = {renames.get(key, key): value for key, value in row.items() if key not in dropped}
                timestamp = (_collection_time(row.get("collection_time", ""))
                             or _collection_time(row.get("DATEADDED", "")) or times.get(_url(row), ""))
                normalized.update(collection_time=timestamp, _source_batch=stamp, _source_row=number)
                writer.writerow(normalized)
                counts["prepared_rows"] += 1
                counts["missing_collection_time"] += not bool(timestamp)
            if (_sha(source) != item["source_sha256"]
                    or (_sha(export) if export.is_file() else None) != item["export_sha256"]):
                raise ValueError("Raw source changed during preparation; use a new output directory.")
    manifest = {"input": item, "articles_sha256": _sha(target), "counts": counts, "excluded": excluded}
    _save(output / "batches" / (stamp + ".json"), manifest)
    return counts


def _check_hash(path, expected):
    if not isinstance(expected, str) or not path.is_file() or _sha(path) != expected:
        raise ValueError("Prepared output integrity/checksum failure: " + str(path))


def _validate_batch(item, output):
    path = output / "batches" / (item["batch"] + ".json")
    manifest = _load(path)
    if manifest.get("input") != item:
        raise ValueError("Batch manifest integrity failure: " + str(path))
    _check_hash(output / "articles" / (item["batch"] + ".csv"), manifest.get("articles_sha256"))
    counts, excluded = manifest.get("counts"), manifest.get("excluded")
    if (not isinstance(counts, dict) or set(counts) != set(_COUNTS)
            or any(type(value) is not int or value < 0 for value in counts.values())
            or counts["input_rows"] != counts["prepared_rows"] + counts["excluded_rows"]
            or counts["missing_collection_time"] > counts["prepared_rows"]
            or not isinstance(excluded, list) or len(excluded) != counts["excluded_rows"]
            or any(not isinstance(row, dict) or set(row) != set(_EXCLUDED_FIELDS)
                   or row["batch"] != item["batch"] for row in excluded)):
        raise ValueError("Batch count/report integrity failure: " + str(path))
    return counts


def _output_layout(output, batches):
    if not output.exists():
        return
    if not output.is_dir():
        raise ValueError("Prepared output must be a directory.")
    stamps = {item["batch"] for item in batches}
    allowed = {"manifest.json", "input_summary.json", "excluded_articles.csv", "articles", "batches"}
    for path in output.rglob("*"):
        _safe_path(path)
        rel = path.relative_to(output)
        if len(rel.parts) == 1 and rel.name in allowed:
            if path.is_dir() == (rel.name in {"articles", "batches"}):
                continue
        if len(rel.parts) == 2 and path.is_file():
            if ((rel.parts[0] == "articles" and path.suffix == ".csv")
                    or (rel.parts[0] == "batches" and path.suffix == ".json")) and path.stem in stamps:
                continue
        raise ValueError("Unowned prepared output; use a new output directory: " + str(path))


def prepare_source_batches(input_dir, prepared_dir, start, end, title_column="title",
                           text_column="fulltext", status_column="extraction_status", success_status="ok",
                           *, progress_fn=None):
    """Return prepared_dir/articles for inclusive, immediate timestamp batches.

    Only successful, nonempty source bodies reach the prepared CSVs. Missing
    GDELT collection timestamps remain blank and are counted, never invented
    from batch-folder names. Completed resumes do not rewrite any output files.
    Changed source/configuration or corrupted outputs require a new directory.
    """
    start, end = _timestamp(start), _timestamp(end)
    if start > end:
        raise ValueError("Start timestamp must not be after end timestamp.")
    settings = {"title_column": title_column, "text_column": text_column,
                "status_column": status_column, "success_status": success_status}
    if any(not isinstance(value, str) or not value.strip() or value != value.strip() for value in settings.values()):
        raise ValueError("Column names and success status must be nonempty strings without surrounding whitespace.")
    if (len({title_column, text_column, status_column}) != 3
            or title_column in {"RAWTEXT", "collection_time", *_RESERVED}
            or text_column in {"TITLE", "collection_time", *_RESERVED}
            or status_column in {"TITLE", "RAWTEXT", "collection_time", *_RESERVED}):
        raise ValueError("Configured source columns conflict with canonical/reserved fields.")
    root, output = _safe_path(input_dir), _safe_path(prepared_dir)
    if not root.is_dir():
        raise ValueError("Source input must be an existing month directory.")
    if root == output or root.is_relative_to(output) or output.is_relative_to(root):
        raise ValueError("Source and prepared directories must be separate and non-overlapping.")
    report = progress_fn or (lambda *args: None)
    batches = _inputs(root, start, end, progress_fn)
    report("Input checks", 0, len(batches))
    _output_layout(output, batches)
    identity = {"version": _VERSION, "source_root": str(root), "start": start, "end": end,
                "columns": settings, "batches": batches}
    fingerprint = _fingerprint(identity)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = _load(manifest_path)
        if manifest.get("input") != identity or manifest.get("fingerprint") != fingerprint:
            raise ValueError("Raw source/range/settings changed; use a new output directory.")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Unowned prepared files; use a new output directory.")
        manifest = {"input": identity, "fingerprint": fingerprint, "completed": False}
    for number, item in enumerate(batches):
        report("Header checks", number, len(batches), item["batch"])
        source = root / item["batch"] / "urls_title.csv"
        with source.open(encoding="utf-8-sig", newline="") as handle:
            _header(csv.DictReader(handle, strict=True), settings)
        report("Header checks", number + 1, len(batches), item["batch"])
    for number, item in enumerate(batches):
        report("Input checks", number, len(batches), item["batch"])
        batch_manifest = output / "batches" / (item["batch"] + ".json")
        if batch_manifest.is_file():
            _validate_batch(item, output)
        elif (output / "articles" / (item["batch"] + ".csv")).exists():
            raise ValueError("Uncommitted snapshot without batch manifest; use a new output directory.")
        elif manifest.get("completed"):
            raise ValueError("Completed preparation is missing a batch; output integrity failure.")
        report("Input checks", number + 1, len(batches), item["batch"])
    if manifest.get("completed"):
        expected_batches = manifest.get("batch_manifests", {})
        if set(expected_batches) != {item["batch"] for item in batches}:
            raise ValueError("Completed batch manifest integrity failure.")
        for number, (stamp, digest) in enumerate(expected_batches.items()):
            report("Manifest checks", number, len(expected_batches), stamp)
            _check_hash(output / "batches" / (stamp + ".json"), digest)
            report("Manifest checks", number + 1, len(expected_batches), stamp)
        for name in ("input_summary.json", "excluded_articles.csv"):
            _check_hash(output / name, manifest.get("outputs", {}).get(name))
        return output / "articles"
    if not manifest_path.exists():
        _save(manifest_path, manifest)
    counts = dict.fromkeys(_COUNTS, 0)
    batch_counts = []
    with _csv_limit():
        try:
            for number, item in enumerate(batches):
                report("Preparing inputs", number, len(batches), item["batch"])
                if (output / "batches" / (item["batch"] + ".json")).is_file():
                    current = _validate_batch(item, output)
                else:
                    current = _prepare_batch(item, root, output, settings)
                batch_counts.append({"batch": item["batch"], **current})
                for name in _COUNTS:
                    counts[name] += current[name]
                report("Preparing inputs", number + 1, len(batches), item["batch"])
        except csv.Error as error:
            raise ValueError("Malformed source/export CSV: " + str(error)) from error
    if _inputs(root, start, end, progress_fn) != batches:
        raise ValueError("Raw batches changed during preparation; use a new output directory.")
    with _atomic_text(output / "excluded_articles.csv") as handle:
        writer = csv.DictWriter(handle, fieldnames=_EXCLUDED_FIELDS, lineterminator="\n")
        writer.writeheader()
        for item in batches:
            writer.writerows(_load(output / "batches" / (item["batch"] + ".json"))["excluded"])
    _save(output / "input_summary.json", {**counts, "selected_batches": [item["batch"] for item in batches],
                                          "batches": batch_counts})
    manifest.update(completed=True,
                    batch_manifests={item["batch"]: _sha(output / "batches" / (item["batch"] + ".json")) for item in batches},
                    outputs={name: _sha(output / name) for name in ("input_summary.json", "excluded_articles.csv")})
    _save(manifest_path, manifest)
    return output / "articles"
