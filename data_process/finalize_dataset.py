"""Step 7: assemble URL-unique records and export the 15 public fields.

Input:
    Country-normalized CSV/CSV.GZ articles, or a flat directory covering all
    intended batches. Source GDELT identifiers and semantic/time/location
    objects are required; database-internal IDs never replace GDELT provenance.
Output:
    One public CSV/CSV.GZ with 15 ordered fields and strict nested JSON. Exact
    duplicate URLs select one coherent article payload and union linked GDELT
    IDs. Reports summary.json, rejected.csv, unresolved.csv and issues.csv
    preserve source rows/reasons without exporting internal helper fields.
Execution:
    Offline, standard-library-only streaming with SQLite-backed deduplication.
    Failed/pending article extractions are not published; geographic uncertainty,
    missing event time and empty location arrays do not remove valid articles.
    Existing outputs require --overwrite; inputs are never overwritten. Exit
    codes are 0 (complete), 2 (review required) and 1 (operational/input error).
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from data_process.utils.process_config import StepArgumentParser, guard_step_selection

import ast
from contextlib import closing, contextmanager
import csv
from datetime import datetime
import gzip
import json
import math
import os
import sqlite3
import tempfile
from urllib.parse import urlsplit


FIELDS = (
    "event_id", "url", "title", "summary", "state_actors", "non_state_actors",
    "individuals", "goldstein_scale_score", "event_importance", "event_sentiment",
    "event_category", "domain", "collection_time", "event_time", "locations",
)
TIME_FIELDS = (
    "time_text", "event_date", "event_time_normalized_start", "event_time_normalized_end",
    "time_precision", "time_basis", "time_is_inferred", "time_confidence",
)
TIME_ALIASES = {
    "event_time_normalized_start": "normalized_start",
    "event_time_normalized_end": "normalized_end",
    "time_is_inferred": "is_inferred", "time_confidence": "confidence",
}
LOCATION_FIELDS = (
    "location_idx", "location_text", "normalized_name", "country", "country_iso3",
    "admin1", "admin2", "location_role", "event_scope", "spatial_precision",
    "location_suitable_for_0_5_degree_grid", "location_suitable_for_country_context",
    "location_confidence", "llm_grid_mappable_flag", "geocode_status", "geocode_provider",
    "lat", "lon", "display_name", "geocode_type", "geocode_confidence", "geocode_success_flag",
)
LOCATION_BOOLEANS = {
    "location_suitable_for_0_5_degree_grid", "location_suitable_for_country_context",
    "llm_grid_mappable_flag", "geocode_success_flag",
}
HISTORICAL_FLAGS = LOCATION_BOOLEANS - {"geocode_success_flag"}
TIME_DEFAULTS = {field: "" for field in TIME_FIELDS}
TIME_DEFAULTS.update(time_precision="unknown", time_basis="unknown",
                     time_confidence="unknown", time_is_inferred=False)
LOCATION_DEFAULTS = {field: "" for field in LOCATION_FIELDS if field not in HISTORICAL_FLAGS}
LOCATION_DEFAULTS.update(location_idx=0, location_role="unknown", event_scope="unknown",
                         spatial_precision="unknown", location_confidence="unknown",
                         geocode_status="unknown", lat=None, lon=None, geocode_success_flag=False)
ID_COLUMNS = ("event_id", "event_ids", "linked_event_ids", "gdelt_event_ids", "GLOBALEVENTID")
URL_COLUMNS = ("url", "URL", "SOURCEURL")
STATUS_COLUMNS = {
    "status", "semantic_status", "relevance_status", "extraction_status", "spacetime_status", "entity_standardization_status",
    "geo_status", "geo_time_status", "geo_extraction_status", "spacetime_extraction_status",
    "time_extraction_status", "location_extraction_status", "geo_temporal_status",
}
SUCCESS_STATUSES = {"ok", "success", "succeeded", "complete", "completed", "done"}
SEMANTIC_LABELS = {
    "event_category": {"Politics", "Economy", "Military", "Diplomacy", "Technology", "Society", "Environment", "Other"},
    "event_sentiment": {"Negative", "Neutral", "Positive", "Other"},
    "event_importance": {"Local", "National", "Regional", "Global", "Other"},
}
REPORT_FIELDS = ("input_file", "input_row", "url", "article_id", "reason", "row_json")


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"Non-standard JSON number: {value}")


def strict_json(value):
    if not isinstance(value, str):
        # Round-tripping also checks direct Python callers for non-finite numbers.
        value = dump(value)
    parsed = json.loads(value, object_pairs_hook=unique_object, parse_constant=reject_constant)
    dump(parsed)  # Reject overflow such as 1e999, even in helper attributes.
    return parsed


def present(value):
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def text(value, field):
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value.strip()


def first(row, columns, default=""):
    for column in columns:
        if present(row.get(column)):
            return row[column]
    return default


def json_shape(value, expected, field):
    parsed = strict_json(value) if present(value) else expected()
    if not isinstance(parsed, expected):
        raise ValueError(f"{field} must be a JSON {expected.__name__}")
    return parsed


def parse_ids(value):
    """Only identifier cells may use the historical Python-literal syntax."""
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            value = strict_json(raw)
        except json.JSONDecodeError:
            if raw.startswith(("[", "(", "'", '"', "{")):
                try:
                    value = ast.literal_eval(raw)
                except (ValueError, SyntaxError) as error:
                    raise ValueError("Malformed event_id") from error
            else:
                value = raw
    values = value if isinstance(value, (list, tuple)) else [value]
    result = set()
    for item in values:
        if type(item) not in (str, int):
            raise ValueError("event_id identifiers must be strings or integers")
        if isinstance(item, str):
            item = item.strip()
            if not item or not item.isascii() or not item.isdecimal():
                raise ValueError("event_id must contain numeric GDELT identifiers, not private article IDs")
            item = int(item)
        if item <= 0:
            raise ValueError("event_id identifiers must be positive")
        result.add(item)
    return sorted(result)


def provenance_ids(row):
    ids = set()
    for column in ID_COLUMNS:
        if present(row.get(column)):
            for item in parse_ids(row[column]):
                ids.add(item)
    if not ids:
        raise ValueError("Missing nonempty event_id provenance")
    return sorted(ids)


def clean_time(value):
    raw = json_shape(value, dict, "event_time")
    result = TIME_DEFAULTS.copy()
    for field in TIME_FIELDS:
        source = field if field in raw else TIME_ALIASES.get(field)
        if source not in raw:
            continue
        item = raw[source]
        if field != "time_is_inferred" and item in (None, ""):
            item = TIME_DEFAULTS[field]
        if field == "time_is_inferred":
            if type(item) is not bool:
                raise ValueError("event_time.time_is_inferred must be boolean")
        elif not isinstance(item, str):
            raise ValueError(f"event_time.{field} must be a string")
        result[field] = item
    return result


def clean_locations(value):
    raw = json_shape(value, list, "locations")
    result = []
    for index, location in enumerate(raw):
        if not isinstance(location, dict):
            raise ValueError("locations must contain objects")
        cleaned = {**LOCATION_DEFAULTS, "location_idx": index}
        for field in LOCATION_FIELDS:
            if field not in location:
                continue
            item = location[field]
            if field not in LOCATION_BOOLEANS and field != "location_idx" and item in (None, ""):
                item = LOCATION_DEFAULTS[field]
            if field in LOCATION_BOOLEANS:
                if type(item) is not bool:
                    raise ValueError(f"locations.{field} must be boolean")
            elif field == "location_idx":
                if type(item) is not int or item < 0:
                    raise ValueError("locations.location_idx must be a nonnegative integer")
            elif field in {"lat", "lon"}:
                bound = 90 if field == "lat" else 180
                if item is not None and (type(item) not in (int, float)
                                         or not math.isfinite(item) or not -bound <= item <= bound):
                    raise ValueError(f"locations.{field} must be null or a finite number in [-{bound}, {bound}]")
            elif not isinstance(item, str):
                raise ValueError(f"locations.{field} must be a string")
            cleaned[field] = item
        if cleaned["geocode_status"] in {"pending", "not_requested"}:
            cleaned.update(geocode_status="unknown", lat=None, lon=None, geocode_success_flag=False)
        result.append(cleaned)
    return result


def clean_entities(value, field):
    values = json_shape(value, list, field)
    result = []
    for item in values:
        if field == "non_state_actors":
            if not isinstance(item, str):
                raise ValueError("non_state_actors must contain strings")
            result.append(item)
            continue
        allowed = ("name", "role") if field == "state_actors" else ("name", "country", "position")
        required = {"name", "role"} if field == "state_actors" else {"name"}
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"{field} must contain objects with {sorted(required)}")
        cleaned = {key: item[key] for key in allowed if key in item}
        if any(not isinstance(v, str) for v in cleaned.values()):
            raise ValueError(f"{field} attributes must be strings")
        result.append(cleaned)
    return result


def clean_score(value):
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError as error:
            raise ValueError("goldstein_scale_score must be numeric text in [-10, 10]") from error
    if type(value) not in (int, float) or not math.isfinite(value) or not -10 <= value <= 10:
        raise ValueError("goldstein_scale_score must be a finite number in [-10, 10]")
    return str(int(value)) if value == int(value) else str(value)


def extraction_problem(row):
    for key, value in row.items():
        name = key.lower().lstrip("_") if isinstance(key, str) else ""
        is_extraction = name in STATUS_COLUMNS or name.endswith("extraction_status")
        if is_extraction and present(value):
            status = text(value, key).lower()
            if status not in SUCCESS_STATUSES:
                return f"Unresolved extraction status {key}={status}"
    return ""


def geographic_problem(row):
    """Geographic uncertainty is reported separately; it does not reject an article."""
    reasons = []
    for field in ("country_normalization_status", "geocoding_status", "geocode_status"):
        status = text(row.get(field), field).lower()
        if status and status not in SUCCESS_STATUSES | {"unknown", "not_requested", "not_found", "non_country", "multi_country"}:
            reasons.append(f"{field}={status}")
    value = first(row, ("locations", "geo_locations_json"), [])
    for index, location in enumerate(json_shape(value, list, "locations")):
        status = location.get("geocode_status", "")
        if status and status not in SUCCESS_STATUSES | {"unknown", "not_requested", "not_found"}:
            reasons.append(f"locations[{index}].geocode_status={status}")
    return "; ".join(reasons)


def canonical_row(row):
    """Normalize a whole candidate; never mix payloads from different rows."""
    result = {field: "" for field in FIELDS}
    result["url"] = text(first(row, URL_COLUMNS), "url")
    if not result["url"]:
        raise ValueError("Missing nonempty URL")
    result["event_id"] = provenance_ids(row)
    inferred = json_shape(row.get("inference"), dict, "inference")
    sections = {}
    for name in ("basic_information", "entities", "sentiment_analysis"):
        section = inferred.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f"inference.{name} must be an object")
        sections[name] = section
    basic, entities, sentiment = (sections[key] for key in sections)
    for field, derived in (
        ("title", first(basic, ("title",), first(row, ("TITLE",)))),
        ("summary", basic.get("summary", "")),
        ("event_category", basic.get("category", "")),
        ("event_sentiment", sentiment.get("event_sentiment", "")),
        ("event_importance", sentiment.get("event_importance_assessment", "")),
    ):
        result[field] = text(first(row, (field,), derived), field)
    for field, allowed in SEMANTIC_LABELS.items():
        if result[field] not in allowed:
            raise ValueError(f"{field} must contain a supported nonempty label")
    score = first(row, ("goldstein_scale_score",), sentiment.get("goldstein_scale_score", ""))
    result["goldstein_scale_score"] = clean_score(score)
    for field in ("state_actors", "non_state_actors", "individuals"):
        result[field] = clean_entities(first(row, (field,), entities.get(field, [])), field)

    # A scalar legacy event_time is collection time; a JSON object is public event time.
    raw_time = row.get("event_time", "")
    public_time = isinstance(raw_time, dict)
    if isinstance(raw_time, str) and raw_time.strip():
        stripped = raw_time.strip()
        public_time = stripped.startswith(("{", "[", '"')) or stripped in {"null", "true", "false", "NaN"}
    legacy_collection = "" if public_time else raw_time
    if present(legacy_collection):
        scalar_time = text(legacy_collection, "legacy event_time")
        try:
            if scalar_time.isdecimal() and len(scalar_time) in (8, 14):
                datetime.strptime(scalar_time, "%Y%m%d" if len(scalar_time) == 8 else "%Y%m%d%H%M%S")
            else:
                datetime.fromisoformat(scalar_time.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("event_time must be a JSON object or a legacy collection timestamp") from error
    result["collection_time"] = text(first(row, ("collection_time", "gdelt_collection_time", "DATEADDED"),
                                           legacy_collection), "collection_time")
    result["event_time"] = clean_time(raw_time if public_time else row.get("geo_event_time_json"))
    result["locations"] = clean_locations(first(row, ("locations", "geo_locations_json"), []))
    domain = first(row, ("domain", "DOMAIN"))
    result["domain"] = text(domain, "domain") if present(domain) else (urlsplit(result["url"]).hostname or "")
    return result


def score_row(row, original):
    """Completeness only: deleted semantic fields and model-suitability flags are irrelevant."""
    scalar_fields = ("title", "summary", "goldstein_scale_score", "event_importance", "event_sentiment", "event_category")
    time = row["event_time"]
    geocoded = [loc for loc in row["locations"]
                if type(loc.get("lat")) in (int, float) and type(loc.get("lon")) in (int, float)]
    raw_length = min(len(text(first(original, ("raw_text", "RAWTEXT")), "raw_text")), 100_000)
    values = (
        int(bool(raw_length)), int(bool(row["title"])),
        sum(bool(row[field]) for field in scalar_fields)
        + sum(bool(row[field]) for field in ("state_actors", "non_state_actors", "individuals")),
        int(any(loc.get("location_role") == "event_site" for loc in geocoded)),
        int(any(loc.get("event_scope") == "primary_event" for loc in geocoded)),
        int(bool(time.get("event_time_normalized_start") or time.get("event_date"))),
        {"low": 1, "medium": 2, "high": 3}.get(time.get("time_confidence", "").lower(), 0),
        int(bool(geocoded)), min(len(geocoded), 99), raw_length,
    )
    return ":".join(f"{value:06d}" for value in values)


def same_path(left, right):
    return left.resolve() == right.resolve() or (
        left.exists() and right.exists() and os.path.samefile(left, right)
    )


def prepare_paths(input_path, output_path, report_dir, chunksize, overwrite):
    if chunksize <= 0:
        raise ValueError("--chunksize must be positive")
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    if not input_path.exists():
        raise ValueError(f"Input does not exist: {input_path}")
    if not output_path.name.lower().endswith((".csv", ".csv.gz")):
        raise ValueError("--output must end with .csv or .csv.gz")
    if input_path.is_dir():
        inputs = sorted(path.resolve() for path in input_path.iterdir()
                        if path.is_file() and path.name.lower().endswith((".csv", ".csv.gz")))
        if not inputs:
            raise ValueError("Input directory contains no CSV or CSV.GZ files")
    else:
        inputs = [input_path]
    report_dir = Path(report_dir).resolve() if report_dir else output_path.parent / (output_path.name + ".reports")
    reports = {name: report_dir / name for name in ("summary.json", "rejected.csv", "unresolved.csv", "issues.csv")}
    targets = [output_path, *reports.values()]
    for index, target in enumerate(targets):
        if any(same_path(target, source) for source in inputs):
            raise ValueError("Input and output/report paths must differ, even with --overwrite")
        if input_path.is_dir() and target.is_relative_to(input_path):
            raise ValueError("Outputs and reports must be outside the input directory")
        if any(same_path(target, other) for other in targets[:index]):
            raise ValueError("Output and report paths must be distinct")
        if target.exists() and (not overwrite or not target.is_file()):
            raise ValueError(f"Output already exists (use --overwrite): {target}")
    return inputs, output_path, reports


@contextmanager
def atomic_text(path, overwrite):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".georel-finalize-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        opener = gzip.open if path.name.lower().endswith(".gz") else open
        with opener(temporary, "wt", encoding="utf-8", newline="") as handle:
            yield handle
        if overwrite:
            os.replace(temporary, path)
        else:
            # An exclusive link avoids a check-then-replace overwrite race.
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def initialize_database(db):
    db.executescript("""
        PRAGMA temp_store=FILE;
        PRAGMA cache_size=-32768;
        CREATE TABLE rows (
            seq INTEGER PRIMARY KEY, source TEXT, source_row INTEGER, url TEXT,
            article_id TEXT, raw TEXT, disposition TEXT, reason TEXT,
            payload TEXT, score TEXT, ids TEXT
        );
        CREATE TABLE article_links (
            article_id TEXT, url TEXT, PRIMARY KEY (article_id, url)
        ) WITHOUT ROWID;
        CREATE TABLE selected (
            url TEXT PRIMARY KEY, payload TEXT, score TEXT
        ) WITHOUT ROWID;
        CREATE TABLE linked_ids (
            url TEXT, id TEXT, PRIMARY KEY (url, id)
        ) WITHOUT ROWID;
    """)


def stage_inputs(db, inputs, chunksize):
    csv.field_size_limit(64 * 1024 * 1024)
    count = 0
    for source in inputs:
        opener = gzip.open if source.name.lower().endswith(".gz") else open
        with opener(source, "rt", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            header = reader.fieldnames
            if not header or len(header) != len(set(header)) or any(not key for key in header):
                raise ValueError(f"Missing, empty or duplicate CSV header in {source}")
            for number, row in enumerate(reader, 2):
                count += 1
                url = text(first(row, URL_COLUMNS), "url")
                article_id = text(row.get("article_id"), "article_id")
                disposition, reason, payload, score, ids = "accepted", "", "", "", "[]"
                if article_id and url:
                    db.execute("INSERT OR IGNORE INTO article_links VALUES (?, ?)", (article_id, url))
                try:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError("CSV row has a different number of cells than the header")
                    reason = extraction_problem(row)
                    if reason:
                        disposition = "unresolved"
                    else:
                        normalized = canonical_row(row)
                        ids = dump(normalized.pop("event_id"))
                        score = score_row(normalized, row)
                        payload = dump(normalized)
                        reason = geographic_problem(row)
                except (ValueError, TypeError, SyntaxError, OverflowError, RecursionError) as error:
                    disposition, reason = "rejected", str(error)
                db.execute("INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (count, str(source), number, url, article_id, dump(row), disposition, reason,
                            payload, score, ids))
                if count % chunksize == 0:
                    db.commit()
    db.commit()
    # Conflicts are found globally before any release output is written.
    db.execute("""
        UPDATE rows SET disposition='rejected',
            reason=CASE WHEN reason='' THEN 'article_id maps to multiple URLs'
                        ELSE reason || '; article_id maps to multiple URLs' END
        WHERE article_id IN (
            SELECT article_id FROM article_links GROUP BY article_id HAVING COUNT(*) > 1
        )
    """)
    db.commit()
    return count


def select_rows(db, chunksize):
    cursor = db.execute("SELECT url, payload, score, ids FROM rows WHERE disposition='accepted' ORDER BY seq")
    for index, (url, payload, score, ids) in enumerate(cursor, 1):
        db.execute("""
            INSERT INTO selected VALUES (?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET payload=excluded.payload, score=excluded.score
            WHERE excluded.score > selected.score
               OR (excluded.score = selected.score AND excluded.payload > selected.payload)
        """, (url, payload, score))
        db.executemany("INSERT OR IGNORE INTO linked_ids VALUES (?, ?)",
                       ((url, dump(item)) for item in strict_json(ids)))
        if index % chunksize == 0:
            db.commit()
    db.commit()


def write_reports(db, reports, summary, overwrite):
    for disposition in ("rejected", "unresolved", "issues"):
        with atomic_text(reports[disposition + ".csv"], overwrite) as handle:
            writer = csv.writer(handle)
            writer.writerow(REPORT_FIELDS)
            predicate = "disposition='accepted' AND reason<>''" if disposition == "issues" else "disposition=?"
            cursor = db.execute("""
                SELECT source, source_row, url, article_id, reason, raw
                FROM rows WHERE """ + predicate + " ORDER BY seq", () if disposition == "issues" else (disposition,))
            writer.writerows(cursor)
    with atomic_text(reports["summary.json"], overwrite) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def finalize_dataset(input_path, output_path, report_dir=None, *, chunksize=10000, overwrite=False):
    inputs, output_path, reports = prepare_paths(input_path, output_path, report_dir, chunksize, overwrite)
    with tempfile.TemporaryDirectory(prefix="georel-finalize-") as scratch:
        with closing(sqlite3.connect(str(Path(scratch) / "articles.sqlite3"))) as db:
            try:
                initialize_database(db)
                input_rows = stage_inputs(db, inputs, chunksize)
                select_rows(db, chunksize)
                counts = dict(db.execute("SELECT disposition, COUNT(*) FROM rows GROUP BY disposition"))
                output_rows = db.execute("SELECT COUNT(*) FROM selected").fetchone()[0]
                rejected, unresolved = counts.get("rejected", 0), counts.get("unresolved", 0)
                geographic_issues = db.execute("SELECT COUNT(*) FROM rows WHERE disposition='accepted' AND reason<>''").fetchone()[0]
                summary = {
                    "status": "incomplete" if rejected or unresolved or geographic_issues else "complete",
                    "input_files": len(inputs), "input_rows": input_rows, "output_rows": output_rows,
                    "accepted_rows": counts.get("accepted", 0), "rejected_rows": rejected,
                    "unresolved_rows": unresolved, "duplicate_rows": counts.get("accepted", 0) - output_rows,
                    "geographic_issue_rows": geographic_issues,
                }
                with atomic_text(output_path, overwrite) as handle:
                    writer = csv.DictWriter(handle, fieldnames=FIELDS)
                    writer.writeheader()
                    for url, payload in db.execute("SELECT url, payload FROM selected ORDER BY url"):
                        row = strict_json(payload)
                        row["event_id"] = [strict_json(item[0]) for item in db.execute(
                            "SELECT id FROM linked_ids WHERE url=? ORDER BY length(id), id", (url,))]
                        for field in ("event_id", "state_actors", "non_state_actors", "individuals", "event_time", "locations"):
                            row[field] = dump(row[field])
                        writer.writerow(row)
                write_reports(db, reports, summary, overwrite)
                return summary
            finally:
                # sqlite's transaction context manager does not close its connection.
                db.close()


def build_parser():
    parser = StepArgumentParser(step="7_finalize_dataset", description=__doc__)
    parser.add_argument("--input", required=True, help="Enriched CSV/CSV.GZ, or a flat directory of CSV files")
    parser.add_argument("--output", required=True, help="Public CSV or CSV.GZ destination")
    parser.add_argument("--report-dir", help="Default: <output filename>.reports alongside the output")
    parser.add_argument("--chunksize", type=int, default=10000, help="Rows per SQLite commit (streamed one row at a time)")
    parser.add_argument("--overwrite", action="store_true", help="Replace outputs/reports, never inputs")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        guard_step_selection(args)
        summary = finalize_dataset(args.input, args.output, args.report_dir,
                                   chunksize=args.chunksize, overwrite=args.overwrite)
    except (OSError, ValueError, csv.Error, sqlite3.Error) as error:
        print(f"Finalization failed: {error}", file=sys.stderr)
        return 1
    print(dump(summary))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
