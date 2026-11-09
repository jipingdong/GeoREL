"""Step 4: extract event time and role-aware locations from source articles.

Input:
    Successful semantic CSV/CSV.GZ articles from Step 3, or a flat directory;
    requires source text, title and URL. Upstream semantic JSON and provenance
    are retained. collection_time is metadata, not the event occurrence time.
Output:
    spatiotemporal_articles.csv and extracted/ carry an event_time JSON object
    and a locations JSON array for geocoding. Empty location arrays retain the
    article and its time. labeled/ and unresolved_articles.csv retain failures;
    stable article IDs prevent row-number joins in downstream processing.
Execution:
    Concurrent deepseek-flash requests (temperature 1), inline prompts,
    parameter/hidden API key, bounded retries and fingerprint-checked resume.
    infer, merge, all and dry-run are supported. No fixed date selection,
    silent text truncation or daily/grid-model geocoding filter is applied.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from data_process.utils.batch_cache import ArticleCache, batch_directory, prepare_cached_batch, write_prediction
from data_process.utils.process_config import StepArgumentParser, guard_step_selection

import hashlib
import json
import math
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import date, datetime
from urllib.parse import urlsplit

import pandas as pd

from data_process.utils.llm_runtime import ArticleProgress, json_hash, make_client as _make_client, safe_error, write_json


SCHEMA_VERSION = "article_spacetime_v3"
SYSTEM_PROMPT = """Extract and normalize the occurrence time and event-related locations of geopolitical-security news using the supplied URL, title, article text and metadata.
Return one JSON object matching the schema below.

Event focus:
- Extract event_time for the central event identified by the headline and lead, not automatically the latest date mentioned.
- For retrospective reporting, the central event may be historical. Do not substitute the reporting date for its occurrence time.
- Extract locations related to the central event and relevant secondary, background or reporting contexts; distinguish these using event_scope.
- Preserve source mentions and supporting evidence. Do not invent missing facts, precision or certainty.

Time normalization:
- collection_time is the GDELT collection timestamp, not the event occurrence time or an authenticated publication timestamp.
- Dates in source_url are weak metadata, not explicit event facts. Do not adopt a URL date as the event date without independent support from the article.
- Resolve expressions such as "yesterday", "Monday" or "last week" only when the article and a reliable reference date support the interpretation.
- Do not assume that the article was published on collection_time. Collection metadata alone is insufficient to establish an exact event date.
- If the year or another date component comes from metadata or contextual inference rather than explicit text, set is_inferred=true.
- Use time_basis="explicit_text" only when the normalized date components are supported by explicit article text.
- Use relative_to_gdelt_collection_time for a relative expression resolved against supported collection metadata; metadata_url_context for supported URL context; model_inference for other contextual inference.
- Metadata-assisted or inferred dates should not receive high confidence unless the article strongly supports the interpretation.
- If normalization remains ambiguous, preserve time_text and evidence, leave unresolved normalized values null, and explain the uncertainty in notes.
- Use ISO 8601 dates/times, YYYY-MM or YYYY at the supported precision. Never manufacture a day from a month/year or invent a timezone.
- Use normalized_end only for a source-supported interval. Do not fill it merely to repeat normalized_start.

Location extraction:
- Preserve raw place mentions. Normalize names and administrative context only when supported and unambiguous.
- Include relevant event sites, directly affected areas, routes, borders, maritime areas, policy implementation areas and reporting locations.
- Do not infer an event location from actor nationality, organisational affiliation or the location of a government headquarters.
- Do not replace a broad place with its capital, centroid or another representative point.
- Retain country-level, broad and uncertain locations at their supported precision; leave unresolved attributes null.
- Avoid duplicate objects for the same place, role and event scope. Separate objects are allowed when distinct roles or scopes are explicitly supported.
- An empty locations array is valid; retain the article's event_time information.

Location roles:
- event_site: where the reported event physically occurred.
- affected_area: an area directly affected by an action, disaster, disruption or policy.
- reporting_location: a dateline, filing or reporting location; not automatically an event site.
- mentioned_context: background, comparison, actor affiliation or indirect contextual reference.
- unknown: the role cannot be determined from the supplied information.

Event scopes and reporting rules:
- primary_event: the central event identified by the headline and lead.
- secondary_event: a separate, directly related event, rather than another description of the central event.
- background_event: prior incidents, historical causes, comparisons or examples used to explain the story.
- reporting_context: datelines, filing locations, source bureaus or publication metadata locations.
- unknown: the relationship to the central event is unclear.
- Do not label a previous-day, previous-week or historical incident primary_event unless it is the article's main focus.
- For a pure dateline such as "BEIJING -", use reporting_location and reporting_context unless the text independently establishes another role.
- For speeches, warnings, threats, diplomatic comments, policy announcements and opinion articles, do not treat the speaker's capital, government seat or reporting bureau as event_site merely because the statement originated there.
- A physical meeting, speech or other event may have an event_site when the article explicitly places it there.
- A territory discussed in a statement may be affected_area only when a direct effect is supported; otherwise use mentioned_context.
- Prefer an explicit unknown or unresolved value to an unsupported interpretation. Explain material ambiguity briefly in notes.

Output schema (all listed keys required; no additional keys):
{
  "event_time": {
    "time_text": null, "normalized_start": null, "normalized_end": null,
    "time_precision": "hour|day|week|month|year|relative|unknown",
    "time_basis": "explicit_text|relative_to_gdelt_collection_time|metadata_url_context|model_inference|unknown",
    "is_inferred": false, "confidence": "high|medium|low", "evidence": null, "notes": null
  },
  "locations": [{
    "location_text": null, "normalized_name": null, "country": null, "admin1": null, "admin2": null,
    "location_role": "event_site|affected_area|reporting_location|mentioned_context|unknown",
    "event_scope": "primary_event|secondary_event|background_event|reporting_context|unknown",
    "spatial_precision": "facility|city|admin2|admin1|subnational_region|country|multi_country_region|maritime_area|border_area|route|unknown",
    "confidence": "high|medium|low", "evidence": null, "notes": null
  }]
}
Choose one value for each controlled label, not a pipe-separated list. Use null for missing text attributes and [] when no locations are supported.
"""
USER_PROMPT = """Extract and normalize the central event's occurrence time and the event-related locations in this news record.
Metadata:
source_url: {url}
collection_time: {collection_time}
Title:
{title}
Source article text:
{raw_text}
"""
TIME_KEYS = {"time_text", "normalized_start", "normalized_end", "time_precision", "time_basis",
             "is_inferred", "confidence", "evidence", "notes"}
LOCATION_KEYS = {"location_text", "normalized_name", "country", "admin1", "admin2", "location_role",
                 "event_scope", "spatial_precision", "confidence", "evidence", "notes"}
ROLES = {"event_site", "affected_area", "reporting_location", "mentioned_context", "unknown"}
SCOPES = {"primary_event", "secondary_event", "background_event", "reporting_context", "unknown"}
PRECISIONS = {"facility", "city", "admin2", "admin1", "subnational_region", "country",
              "multi_country_region", "maritime_area", "border_area", "route", "unknown"}


@dataclass
class Settings:
    api_key: str | None = field(default=None, repr=False)
    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    temperature: float = 1.0
    workers: int = 10
    attempts: int = 3
    timeout: float = 180
    retry_delay: float = 1
    cache_root: Path | None = None
    legacy_cache_roots: tuple[Path, ...] = ()


@dataclass
class Batch:
    source: Path
    data: pd.DataFrame
    directory: Path
    res_dir: Path
    manifest: dict
    article_cache: ArticleCache | None = None


@dataclass(frozen=True)
class _BatchPlan:
    source: Path
    directory: Path | None
    manifest: dict
    cached: int | None
    columns: tuple[str, ...]


def _plan_batch(batch, check_cache=True):
    return _BatchPlan(batch.source, batch.directory if check_cache else None, dict(batch.manifest),
                      sum(load_prediction(batch, i)["status"] == "ok" for i in range(len(batch.data))) if check_cache else None,
                      tuple(batch.data.columns))


def _load_batches(plans, output, settings, *, merge_only=True, restore_cache=True):
    for number, plan in enumerate(plans):
        batch = prepare_batch(plan.source, output, settings, merge_only=merge_only,
                              restore_cache=restore_cache)
        changed = batch.manifest["source_sha256"] != plan.manifest["source_sha256"]
        if not merge_only:
            changed = changed or batch.manifest != plan.manifest
        if plan.directory is not None:
            changed = changed or batch.manifest != plan.manifest or batch.directory != plan.directory
        if changed:
            raise ValueError(f"Input changed after preflight: {plan.source.name}; rerun with stable inputs.")
        if not merge_only:
            plans[number] = _BatchPlan(batch.source, batch.directory, dict(batch.manifest),
                                       None, tuple(batch.data.columns))
        yield batch


def _unique_object(pairs):
    if len(pairs) != len({k for k, _ in pairs}):
        raise ValueError("Duplicate JSON keys are not accepted.")
    return dict(pairs)


def _reject_constant(value):
    raise ValueError("Non-finite JSON numbers are not accepted.")


def _date_value(value):
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError("Normalized dates must be strings or null.")
    try:
        if re.fullmatch(r"\d{4}", value):
            return date(int(value), 1, 1)
        if re.fullmatch(r"\d{4}-\d{2}", value):
            return date.fromisoformat(value + "-01")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.date()
    except ValueError as error:
        raise ValueError("Invalid normalized calendar date/time.") from error


def validate_inference(obj):
    if not isinstance(obj, dict) or set(obj) != {"event_time", "locations"}:
        raise ValueError("Expected event_time and locations, not an error or partial response.")
    event = obj["event_time"]
    if not isinstance(event, dict) or set(event) != TIME_KEYS or not isinstance(obj["locations"], list):
        raise ValueError("Incorrect time/location schema.")
    if type(event["is_inferred"]) is not bool:
        raise ValueError("is_inferred must be a JSON boolean.")
    if any(v is not None and not isinstance(v, str) for k, v in event.items() if k != "is_inferred"):
        raise ValueError("Time text attributes must be strings or null.")
    if event["time_precision"] not in {"hour", "day", "week", "month", "year", "relative", "unknown"}:
        raise ValueError("Unknown time precision.")
    if event["time_basis"] not in {"explicit_text", "relative_to_gdelt_collection_time",
                                   "metadata_url_context", "model_inference", "unknown"}:
        raise ValueError("Unknown time basis.")
    start, end = _date_value(event["normalized_start"]), _date_value(event["normalized_end"])
    if start and end and end < start:
        raise ValueError("Event end precedes its start.")
    if start and end and "T" in event["normalized_start"] and "T" in event["normalized_end"]:
        start_ts = datetime.fromisoformat(event["normalized_start"].replace("Z", "+00:00"))
        end_ts = datetime.fromisoformat(event["normalized_end"].replace("Z", "+00:00"))
        if (start_ts.tzinfo is None) == (end_ts.tzinfo is None) and end_ts < start_ts:
            raise ValueError("Event end precedes its start.")
    if event["confidence"] not in {"high", "medium", "low"}:
        raise ValueError("Unknown time confidence.")
    for loc in obj["locations"]:
        if not isinstance(loc, dict) or set(loc) != LOCATION_KEYS:
            raise ValueError("Incorrect location schema.")
        if any(v is not None and not isinstance(v, str) for v in loc.values()):
            raise ValueError("Location attributes must be strings or null.")
        if loc["location_role"] not in ROLES or loc["event_scope"] not in SCOPES:
            raise ValueError("Unknown location role or event scope.")
        if loc["spatial_precision"] not in PRECISIONS or loc["confidence"] not in {"high", "medium", "low"}:
            raise ValueError("Unknown location precision/confidence.")
    return obj


def parse_prediction(raw):
    if not isinstance(raw, str):
        raise ValueError("Model returned no JSON text.")
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("Incomplete JSON block.")
        raw = "\n".join(lines[1:-1])
    return validate_inference(json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant))


def normalize_source(path):
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    reserved = {"spacetime_status", "spacetime_error", "locations", "geo_event_time_json", "geo_locations_json"}
    if reserved.intersection(data.columns):
        raise ValueError("Use upstream source/semantic articles, not an already enriched table.")
    for target, aliases in (("TITLE", ("title",)), ("RAWTEXT", ("raw_text",)),
                            ("url", ("SOURCEURL", "source_url")),
                            ("collection_time", ("gdelt_collection_time", "DATEADDED", "event_time"))):
        if target not in data.columns:
            alias = next((a for a in aliases if a in data.columns), None)
            if alias:
                data[target] = data[alias]
    if not {"TITLE", "RAWTEXT", "url"}.issubset(data.columns):
        raise ValueError("Input needs TITLE/RAWTEXT/SOURCEURL or title/raw_text/url columns.")
    if data["url"].str.strip().eq("").any() or data["RAWTEXT"].str.strip().eq("").any():
        raise ValueError("Each source article needs a URL and nonempty source text.")
    if "collection_time" not in data:
        data["collection_time"] = ""
    if "event_time" in data:
        if data["event_time"].str.lstrip().str.startswith(("{", "[")).any():
            raise ValueError("event_time is already a structured field; use the upstream source table.")
        data = data.drop(columns=["event_time"])
    for column in ("semantic_status", "relevance_status", "entity_standardization_status"):
        if column in data and not data[column].eq("ok").all():
            raise ValueError(f"Input contains unresolved {column}; use the previous successful output.")
    if "article_id" not in data:
        # Stage provenance distinguishes repeated URLs until final global deduplication.
        columns = [c for c in ("url", "collection_time", "TITLE", "RAWTEXT", "event_id", "GLOBALEVENTID",
            "event_ids", "linked_event_ids", "gdelt_event_ids", "_id", "_source_batch", "_source_row") if c in data]
        occurrences, identities = {}, []
        for row in data.to_dict("records"):
            base = json_hash({"source_batch": Path(path).name, "article": {c: row[c] for c in columns}})
            occurrences[base] = occurrences.get(base, 0) + 1
            identities.append(json_hash({"source_identity": base, "occurrence": occurrences[base]}))
        data["article_id"] = pd.Series(identities, dtype=str)
    if data["article_id"].str.strip().eq("").any() or data["article_id"].duplicated().any():
        raise ValueError("article_id must be nonempty and unique within each input batch.")
    return data.reset_index(drop=True)


def prepare_batch(source, output, settings, merge_only=False, *, inspect_only=False, restore_cache=True):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Input and output must not overlap.")
    data = normalize_source(source)
    name = source.name.removesuffix(".gz").removesuffix(".csv")
    directory = batch_directory(output, name, settings)
    source_hash = hashlib.sha256(data.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()
    prompt_hash = json_hash([SYSTEM_PROMPT, USER_PROMPT])
    metadata = {"stage": "spacetime", "schema_version": SCHEMA_VERSION, "source_sha256": source_hash,
                "prompt_sha256": prompt_hash, "model": settings.model,
                "base_url": settings.base_url.rstrip("/"), "temperature": float(settings.temperature)}
    manifest = {**metadata, "fingerprint": json_hash(metadata), "source_file": source.name, "rows": len(data)}
    batch = Batch(source, data, directory, directory / "res", manifest)
    if inspect_only:
        return batch
    return prepare_cached_batch(batch, settings, load_prediction, merge_only=merge_only,
                                stable_keys=("source_sha256", "prompt_sha256", "schema_version"),
                                content=_request_content, validate=lambda payload, row: validate_inference(payload["inference"]),
                                restore_cache=restore_cache)


def _request_content(row):
    return USER_PROMPT.format(url=row["url"], collection_time=row["collection_time"],
                              title=row["TITLE"], raw_text=row["RAWTEXT"])


def _atomic_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, obj)
    temporary.replace(path)


def load_prediction(batch, index):
    path = batch.res_dir / f"{index}.json"
    if not path.exists():
        return {"status": "pending", "inference": None, "error": ""}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                         parse_constant=_reject_constant)
        if (obj.get("fingerprint") != batch.manifest["fingerprint"] or type(obj.get("idx")) is not int
                or obj["idx"] != index or obj.get("article_id") != batch.data.iloc[index]["article_id"]):
            raise ValueError("Cache does not match article/input/configuration.")
        if obj.get("status") == "ok":
            return {"status": "ok", "inference": validate_inference(obj.get("inference")), "error": ""}
        if obj.get("status") == "failed":
            return {"status": "failed", "inference": None, "error": str(obj.get("error", "Request failed."))}
        raise ValueError("Unknown cached status.")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        return {"status": "failed", "inference": None, "error": f"Invalid cache: {error}"}


def _extract_article(batch, index, settings, request):
    row = batch.data.iloc[index]
    content = _request_content(row)
    result = {"idx": index, "article_id": row["article_id"], "fingerprint": batch.manifest["fingerprint"]}
    for attempt in range(1, settings.attempts + 1):
        try:
            obj = parse_prediction(request(content))
            result.pop("error", None)
            result.update(status="ok", inference=obj, attempts=attempt)
            break
        except Exception as error:
            result.update(status="failed", error=safe_error(error, settings.api_key), attempts=attempt)
            if attempt < settings.attempts:
                time.sleep(settings.retry_delay * 2 ** (attempt - 1))
    write_prediction(batch, index, result)
    return result["status"]


def _run_inference_batches(batches, settings, request=None, *, total, cached_counts, progress=None):
    """Bound submissions across files without retaining the selected corpus."""
    totals = {"processed": 0, "reused": sum(cached_counts.values()), "success": 0, "failed": 0}
    owns_progress = progress is None
    if owns_progress:
        progress = ArticleProgress("Spatiotemporal", workers=settings.workers)

    def tasks():
        for batch in batches:
            pending = [i for i in range(len(batch.data)) if load_prediction(batch, i)["status"] != "ok"]
            reused = len(batch.data) - len(pending)
            if reused != cached_counts.get(batch.source, 0):
                totals["reused"] += reused - cached_counts.get(batch.source, 0)
                progress.update(totals, active=len(futures))
            for index in pending:
                yield batch, index

    client = None
    def api_request(content):
        response = client.chat.completions.create(
            model=settings.model, temperature=settings.temperature, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
            stream=False)
        return response.choices[0].message.content

    pending_tasks = tasks()
    try:
        progress.plan(total, totals["reused"])
        with ThreadPoolExecutor(max_workers=settings.workers) as executor:
            futures = {}
            def fill_queue():
                nonlocal client
                while len(futures) < settings.workers * 2:
                    task = next(pending_tasks, None)
                    if task is None:
                        break
                    batch, index = task
                    if request is None and client is None:
                        client = _make_client(settings)
                    future = executor.submit(_extract_article, batch, index, settings,
                                             request if request is not None else api_request)
                    futures[future] = batch.source
            try:
                fill_queue()
                progress.update(totals, active=len(futures))
                while futures:
                    done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                    for future in done:
                        status = future.result()
                        futures.pop(future)
                        totals["processed"] += 1
                        totals["success" if status == "ok" else "failed"] += 1
                    fill_queue()
                    progress.update(totals, active=len(futures))
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    finally:
        try:
            pending_tasks.close()
            if client is not None:
                client.close()
        finally:
            if owns_progress:
                progress.close()
    return totals


def run_inference(batch, settings, request=None):
    """Retain the single-batch API with the same cache/retry implementation."""
    cached = sum(load_prediction(batch, i)["status"] == "ok" for i in range(len(batch.data)))
    counts = _run_inference_batches((batch,), settings, request,
                                    total=len(batch.data), cached_counts={batch.source: cached})
    return {key: counts[key] for key in ("processed", "reused")}


def public_objects(obj):
    event = obj["event_time"]
    text = lambda value: "" if value is None else value
    event_time = {"time_text": text(event["time_text"]),
                  "event_date": text(event["normalized_start"])[:10],
                  "event_time_normalized_start": text(event["normalized_start"]),
                  "event_time_normalized_end": text(event["normalized_end"]),
                  "time_precision": event["time_precision"], "time_basis": event["time_basis"],
                  "time_is_inferred": event["is_inferred"], "time_confidence": event["confidence"]}
    locations = []
    for index, loc in enumerate(obj["locations"]):
        item = {k: text(v) for k, v in loc.items() if k != "confidence"}
        item.update(location_idx=index, location_confidence=loc["confidence"])
        locations.append(item)
    return event_time, locations


def merge_batches(batches, output, merged_name="spatiotemporal_articles.csv", columns=None):
    output = Path(output)
    if columns is None:
        batches = list(batches)
        columns = list(dict.fromkeys(c for batch in batches for c in batch.data.columns))
    columns = list(dict.fromkeys([*columns, "event_time", "locations", "spacetime_status", "spacetime_error",
                                 "_spacetime_input_file", "_spacetime_input_row"]))
    totals = {"total": 0, "success": 0, "failed": 0, "pending": 0}
    first = True
    for batch in batches:
        results = [load_prediction(batch, i) for i in range(len(batch.data))]
        frame = batch.data.copy()
        objects = [public_objects(r["inference"]) if r["status"] == "ok" else (None, None) for r in results]
        for column, pos in (("event_time", 0), ("locations", 1)):
            frame[column] = [json.dumps(obj[pos], ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                             if obj[pos] is not None else "" for obj in objects]
        frame["spacetime_status"] = [r["status"] for r in results]
        frame["spacetime_error"] = [r["error"] for r in results]
        keep = frame["spacetime_status"].eq("ok")
        for name, rows in (("labeled", frame), ("extracted", frame.loc[keep])):
            (output / name).mkdir(parents=True, exist_ok=True)
            rows.to_csv(output / name / batch.source.name, index=False, encoding="utf-8")
        frame["_spacetime_input_file"] = batch.source.name
        frame["_spacetime_input_row"] = range(len(frame))
        frame = frame.reindex(columns=columns)
        for name, rows in ((merged_name, frame.loc[keep]), ("unresolved_articles.csv", frame.loc[~keep])):
            rows.to_csv(output / name, mode="w" if first else "a", header=first, index=False, encoding="utf-8")
        first = False
        totals["total"] += len(frame)
        totals["success"] += int(keep.sum())
        for status in ("failed", "pending"):
            totals[status] += sum(r["status"] == status for r in results)
    totals["status"] = "incomplete" if totals["failed"] + totals["pending"] else "complete"
    _atomic_json(output / "run_summary.json", totals)
    return totals


def build_parser():
    parser = StepArgumentParser(step="4_spatiotemporal_extraction", description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("infer", "merge", "all"), default="all")
    parser.add_argument("--api-key", help="Runtime key; omit for hidden input. Never saved or read from environment.")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--attempts", type=int, default=3, help="Total attempts, including the first request.")
    parser.add_argument("--retry-delay", type=float, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--merged-name", default="spatiotemporal_articles.csv")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    settings = Settings(api_key=args.api_key, model=args.model, base_url=args.base_url.rstrip("/"),
                        temperature=args.temperature, workers=args.workers, attempts=args.attempts,
                        timeout=args.timeout, retry_delay=args.retry_delay,
                        cache_root=getattr(args, "shared_cache_dir", None),
                        legacy_cache_roots=getattr(args, "legacy_cache_roots", ()))
    progress = None
    try:
        endpoint = urlsplit(settings.base_url)
        if (endpoint.scheme not in {"http", "https"} or not endpoint.hostname or endpoint.username
                or endpoint.password or endpoint.query or endpoint.fragment):
            raise ValueError("base-url must be an HTTP(S) endpoint without credentials, query or fragment.")
        if (settings.workers < 1 or settings.attempts < 1 or not settings.model.strip()
                or not math.isfinite(settings.temperature) or not 0 <= settings.temperature <= 2
                or not math.isfinite(settings.timeout) or settings.timeout <= 0
                or not math.isfinite(settings.retry_delay) or settings.retry_delay < 0):
            raise ValueError("Invalid model, temperature, worker, attempt, timeout or delay setting.")
        if not args.dry_run and args.stage in {"infer", "all"}:
            progress = ArticleProgress("Spatiotemporal", workers=settings.workers)
        source, output = args.input.resolve(), args.output_dir.resolve()
        if source == output or source.is_relative_to(output) or output.is_relative_to(source):
            raise ValueError("Input and output must not overlap.")
        if args.dry_run and args.stage == "merge":
            raise ValueError("dry-run cannot be combined with stage=merge.")
        if Path(args.merged_name).name != args.merged_name or not args.merged_name.endswith(".csv") or args.merged_name == "unresolved_articles.csv":
            raise ValueError("merged-name must be a simple CSV filename other than unresolved_articles.csv.")
        files = [source] if source.is_file() else sorted([*source.glob("*.csv"), *source.glob("*.csv.gz")]) if source.is_dir() else []
        if not files or any(not p.name.endswith((".csv", ".csv.gz")) for p in files):
            raise ValueError("Input must be a CSV/CSV.GZ file or a flat directory of CSV files.")
        names = [p.name.removesuffix(".gz").removesuffix(".csv") for p in files]
        if len(set(names)) != len(names):
            raise ValueError("Input CSV/GZ batch names must be distinct.")
        guard_step_selection(args)
        records = unresolved = cached_successes = 0
        plans = []
        # Count and validate source rows without opening per-article caches.
        for number, path in enumerate(files):
            if progress is not None:
                progress.preparing("Input counts", number, len(files), path.name,
                                   records=records)
            batch = prepare_batch(path, output, settings, merge_only=args.stage == "merge",
                                  inspect_only=not args.dry_run)
            plan = _plan_batch(batch, check_cache=args.dry_run)
            plans.append(plan)
            records += len(batch.data)
            cached_successes += plan.cached or 0
            if progress is not None:
                progress.preparing("Input counts", number + 1, len(files), path.name,
                                   records=records)
        batch = None
        if args.dry_run:
            print(json.dumps({"stage": "dry-run", "batches": len(files), "records": records,
                              "cached_successes": cached_successes, "requests_needed": records - cached_successes}))
            return 0
        if args.stage in {"infer", "all"}:
            counts = _run_inference_batches(_load_batches(plans, output, settings, merge_only=False), settings,
                                            total=records, cached_counts={},
                                            progress=progress)
            unresolved = counts["failed"]
        if args.stage in {"merge", "all"}:
            columns = list(dict.fromkeys(c for plan in plans for c in plan.columns))
            batches = _load_batches(plans, output, settings, restore_cache=args.stage == "merge")
            summary = merge_batches(batches, output, args.merged_name, columns)
        else:
            summary = {"status": "incomplete" if unresolved else "complete", "unresolved": unresolved}
        if progress is not None:
            progress.close()
        print(json.dumps(summary, indent=2))
        return 0 if summary["status"] == "complete" else 2
    except Exception as error:
        if progress is not None:
            progress.close()
        print(f"Error: {safe_error(error, settings.api_key)}")
        return 1
    finally:
        if progress is not None:
            progress.close()


if __name__ == "__main__":
    raise SystemExit(main())
