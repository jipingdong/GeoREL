"""Step 3: extract and validate article-level event semantics.

Input:
    Successfully relevant article CSVs from relevance_filter, containing TITLE,
    RAWTEXT and the source/provenance columns needed by downstream stages.
Output:
    extract_res.csv and extracted/ contain successful strict-JSON inference
    objects. labeled/ and unresolved_articles.csv preserve failed/pending rows.
    Semantic fields cover summaries, controlled categories, actors, individuals,
    sentiment, importance and one Goldstein-style score; structured time/place
    objects are extracted separately in Step 4.
Execution:
    Cross-file deepseek-flash requests (temperature 1) share one global worker
    pool and API client. Inputs are preflighted, then loaded lazily into a bounded
    queue. Parameter/hidden API key, bounded retries and article-cache resume
    are retained. Eight actor roles and all
    classification enums are validated before success is cached. infer, merge,
    all and dry-run follow the relevance-stage conventions. Pass successful
    articles directly to spatiotemporal_extraction in Step 4.
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
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import pandas as pd

from data_process.utils.llm_runtime import ArticleProgress, collect_inputs, json_hash, make_client as _make_client, safe_error, write_json

SYSTEM_PROMPT = """
You are a geopolitical security and international affairs analysis expert. Your task is to perform structured information extraction and event analysis on the provided news text. 
Please strictly follow the instructions below and present the output in standard JSON format.

### Basic Information Extraction

1.  **Title**: Extract the original news title. If there is no clear title or the title is incomplete, generate one based on the content.
2.  **Summary**: Summarize the core content of the news in approximately 100-150 words, including the 5W1H elements (Who/What/When/Where/Why/How).
3.  **Category**: Select the most matching tag from [Politics|Economy|Military|Diplomacy|Technology|Society|Environment|Other].
    - **Politics**: Governance, elections, domestic policy, or political power.
    - **Economy**: Trade, finance, business, markets, or economic policy.
    - **Military**: Armed conflict, military operations, defense, or weapons.
    - **Diplomacy**: Official international relations, negotiations, or agreements.
    - **Technology**: Scientific or technological development, deployment, or policy.
    - **Society**: Social issues, public health, education, culture, or community life.
    - **Environment**: Climate, ecosystems, pollution, conservation, or natural hazards.
    - **Other**: A source-supported topic outside the seven categories above; not a default for unknown information.

### Key Entity Recognition

1.  **State Actors**: Extract countries involved in the focal event, each with exactly one nonempty, best source-supported role from [Initiator|Target|Participant|Mediator|Observer|Host|Supporter|Opponent].
    - **Initiator**: Starts or carries out the focal action, rather than merely taking part.
    - **Target**: The direct recipient of the focal action (e.g., an attack, sanction, demand, or aid), not merely a participant, host, or indirectly affected country.
    - **Participant**: Actively takes part in a joint event or activity without a more specific source-supported role; attendance alone does not make a country a Target.
    - **Mediator**: Intervenes between parties to facilitate dialogue, negotiation, or settlement.
    - **Observer**: Explicitly monitors or observes without actively taking part in the focal action.
    - **Host**: Explicitly convenes or provides the venue for the event; location alone does not establish hosting or make the host a Target.
    - **Supporter**: Explicitly endorses or assists a party, policy, or action without itself carrying out the focal action.
    - **Opponent**: Explicitly opposes or resists a party, policy, or action; this alone does not establish receipt of the focal action as a Target.
    Choose the most specific role supported by the article. If no role is adequately supported, omit the actor; do not invent actors or roles, use an empty role, or use Participant as an unknown-role fallback. Affected Party is not an automatic alias for Target: indirect effects alone do not establish direct receipt of an action.
2.  **Non-State Actors**: Extract international organizations (e.g., UN/NATO), armed groups, political parties, NGOs, companies, and other institutions (e.g., think tanks) involved in the event. Return names as strings, not role objects.
3.  **Individuals:** Extract key persons involved in the event, including their names, positions, and their country of affiliation.
    For all entities, use the canonical full name when explicitly provided in the text; otherwise preserve the source name and do not invent aliases, acronym expansions, affiliations, or entities.

### News Sentiment Analysis

1.  **Event Sentiment**: Select the most matching tag from [Negative|Neutral|Positive|Other], judging the event rather than the wording of the report.
    - **Negative**: Predominantly conflictual, harmful, or deteriorating developments.
    - **Neutral**: Predominantly descriptive or routine developments without clear positive or negative direction.
    - **Positive**: Predominantly cooperative, beneficial, or improving developments.
    - **Other**: Explicitly mixed or otherwise evidenced sentiment that no single label above captures; not a default for unknown sentiment.
2.  **Event Importance Assessment**: Select the best source-supported scope from [Local|National|Regional|Global|Other].
    - **Local**: Primarily affects a locality or subnational area.
    - **National**: Primarily affects one country at a national level.
    - **Regional**: Primarily affects several countries within a geographic region.
    - **Global**: Has worldwide reach or implications beyond a single region; foreign involvement alone is insufficient.
    - **Other**: An explicitly evidenced scope outside these geographic levels, such as a non-geographic community; not a default for unknown scope.
3.  **Goldstein Scale Score**: A quantitative indicator of the event's positive or negative intensity. Return one finite number in [-10, 10], or null or an empty string ("") if the text gives insufficient evidence. Do not return booleans, nonfinite numbers, numeric strings, ranges, lists, or multiple scores. For example: -10 Large-scale war/invasion; -5 Military conflict/attack; 0 Neutral event/statement; +5 Agreement signing/cooperation statement; +10 Full alliance/peace treaty. Keep the score coherent with the event and its sentiment; do not infer successful cooperation from talks alone.

### Output Requirements

1.  Present the output in **JSON format**, using only the fields shown below.
2.  Missing free-text attributes use an empty string (""); missing entities use empty arrays ([]), not blank-name placeholders. Every included entity must have a nonempty name. Category, event_sentiment, and event_importance_assessment must each be one nonempty label from their respective lists, chosen from the available source evidence. Other is not a universal missing-value filler. State-actor objects must have one of the eight nonempty roles above; unsupported or ambiguous roles are invalid, not aliases to be silently converted.

**Illustrative Example Output (fictional, not a factual report):**

```json
{
  "basic_information": {
    "title": "U.S. and China Hold Strategic Stability Talks",
    "summary": "The United States and China held strategic stability talks in Geneva aimed at discussing tensions in the Taiwan Strait. Officials from both countries took part in the meeting and outlined their positions on avoiding misjudgment and escalation of conflict. The discussion concerned security issues with implications for several countries in the surrounding region. The report described the exchange of views but did not announce a signed agreement, a settlement, or any concrete change in either side's policies. It did not identify individual attendees or participating organizations. The event therefore represents an ongoing diplomatic discussion, with no demonstrated positive or negative outcome beyond the meeting itself.",
    "category": "Diplomacy"
  },
  "entities": {
    "state_actors": [
      {
        "name": "United States",
        "role": "Participant"
      },
      {
        "name": "China",
        "role": "Participant"
      }
    ],
    "non_state_actors": [],
    "individuals": []
  },
  "sentiment_analysis": {
    "event_sentiment": "Neutral",
    "event_importance_assessment": "Regional",
    "goldstein_scale_score": 0
  }
}
"""

USER_PROMPT = """
Below is the news text:
"""

SCHEMA_VERSION = "semantic_core_v3"
STATE_ACTOR_ROLES = {"Initiator", "Target", "Participant", "Mediator", "Observer", "Host", "Supporter", "Opponent"}
CATEGORIES = {"Politics", "Economy", "Military", "Diplomacy", "Technology", "Society", "Environment", "Other"}
SENTIMENTS = {"Negative", "Neutral", "Positive", "Other"}
IMPORTANCE = {"Local", "National", "Regional", "Global", "Other"}


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


def _object(value, required, allowed=None):
    allowed = set(required) if allowed is None else set(allowed)
    if not isinstance(value, dict) or not set(required).issubset(value) or set(value) - allowed:
        raise ValueError("Missing, unexpected or incorrectly structured extraction fields.")


def _strings(value, required, allowed=None):
    _object(value, required, allowed)
    if any(not isinstance(v, str) for v in value.values()):
        raise ValueError("Text attributes must be strings; use an empty string for missing text.")


def validate_inference(value):
    _object(value, {"basic_information", "entities", "sentiment_analysis"})
    basic, entities, sentiment = value["basic_information"], value["entities"], value["sentiment_analysis"]
    _strings(basic, {"title", "summary", "category"})
    if basic["category"] not in CATEGORIES:
        raise ValueError("Unknown event category.")
    _object(entities, {"state_actors", "non_state_actors", "individuals"})
    if any(not isinstance(v, list) for v in entities.values()):
        raise ValueError("Entity fields must be arrays.")
    for actor in entities["state_actors"]:
        _strings(actor, {"name", "role"})
        if not actor["name"].strip():
            raise ValueError("Included state actors require nonempty names; use [] for missing entities.")
        if actor["role"] not in STATE_ACTOR_ROLES:
            raise ValueError("State-actor role must be one of the eight nonempty controlled labels.")
    if any(not isinstance(name, str) or not name.strip() for name in entities["non_state_actors"]):
        raise ValueError("Non-state actors must be nonempty names, not objects or blank placeholders.")
    for person in entities["individuals"]:
        _strings(person, {"name"}, {"name", "country", "position"})
        if not person["name"].strip():
            raise ValueError("Included individuals require nonempty names; use [] for missing entities.")
    _object(sentiment, {"event_sentiment", "event_importance_assessment", "goldstein_scale_score"})
    if not isinstance(sentiment["event_sentiment"], str) or sentiment["event_sentiment"] not in SENTIMENTS:
        raise ValueError("Unknown or non-text event sentiment.")
    if not isinstance(sentiment["event_importance_assessment"], str) or sentiment["event_importance_assessment"] not in IMPORTANCE:
        raise ValueError("Unknown or non-text event importance.")
    score = sentiment["goldstein_scale_score"]
    if score is not None and score != "":
        if type(score) not in (int, float) or not math.isfinite(score) or not -10 <= score <= 10:
            raise ValueError("Goldstein-style score must be one finite number from -10 to 10, or missing.")
    return value


def _unique_object(pairs):
    if len(pairs) != len({k for k, _ in pairs}):
        raise ValueError("Duplicate JSON keys are not accepted.")
    return dict(pairs)


def _reject_constant(value):
    raise ValueError("Non-standard JSON numbers are not accepted.")


def parse_prediction(raw):
    if not isinstance(raw, str):
        raise ValueError("Model returned no JSON text.")
    raw = raw.strip()
    fence = chr(96) * 3
    if raw.startswith(fence):
        lines = raw.splitlines()
        if len(lines) < 3 or lines[-1].strip() != fence:
            raise ValueError("Incomplete JSON code block.")
        raw = "\n".join(lines[1:-1])
    try:
        obj = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, TypeError) as error:
        raise ValueError("Model response is not an unambiguous, standard JSON object.") from error
    return validate_inference(obj)


def normalize_source(path):
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"TITLE", "RAWTEXT"}.issubset(data.columns):
        raise ValueError(f"{Path(path).name} must contain TITLE and RAWTEXT.")
    reserved = {"inference", "semantic_status", "semantic_error", "_semantic_input_file", "_semantic_input_row"}
    if reserved.intersection(data.columns):
        raise ValueError("Use relevant input articles, not this script's extraction output.")
    if "relevance" in data and not data["relevance"].str.strip().str.lower().eq("true").all():
        raise ValueError("Input must contain only successfully relevant articles; select relevance_filter's relevant output.")
    if "relevance_status" in data and not data["relevance_status"].eq("ok").all():
        raise ValueError("Input contains failed or pending relevance labels.")
    # Do not deduplicate or rewrite upstream article/provenance columns here.
    return data.reset_index(drop=True)


def prepare_batch(source, output, settings, merge_only=False, *, inspect_only=False, restore_cache=True):
    source, output = Path(source).resolve(), Path(output).resolve()
    data = normalize_source(source)
    directory = batch_directory(output, source.stem, settings)
    source_hash = hashlib.sha256(data.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()
    prompt_hash = json_hash([SYSTEM_PROMPT, USER_PROMPT])
    metadata = {"stage": "semantic", "schema_version": SCHEMA_VERSION, "source_sha256": source_hash,
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
    return USER_PROMPT + f"\nTitle: {row['TITLE']} \n\n News text: {row['RAWTEXT']}"


def load_prediction(batch, index):
    path = batch.res_dir / f"{index}.json"
    if not path.exists():
        return {"status": "pending", "inference": None, "error": ""}
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("fingerprint") != batch.manifest["fingerprint"] or type(item.get("idx")) is not int or item["idx"] != index:
            raise ValueError("Cached result does not match this input/configuration.")
        if item.get("status") == "ok":
            return {"status": "ok", "inference": validate_inference(item.get("inference")), "error": ""}
        if item.get("status") == "failed":
            return {"status": "failed", "inference": None, "error": str(item.get("error", "Request failed."))}
        raise ValueError("Unknown cached result status.")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        return {"status": "failed", "inference": None, "error": f"Invalid cache: {error}"}


def _extract_article(batch, index, settings, request):
    row = batch.data.iloc[index]
    content = _request_content(row)
    result = {"idx": index, "fingerprint": batch.manifest["fingerprint"]}
    for attempt in range(1, settings.attempts + 1):
        try:
            inference = parse_prediction(request(content))
            result.pop("error", None)
            result.update(status="ok", inference=inference, attempts=attempt)
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
        progress = ArticleProgress("Semantic", workers=settings.workers)

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
            model=settings.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
            temperature=settings.temperature, response_format={"type": "json_object"}, stream=False)
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


def merge_batches(batches, output, merged_name="extract_res.csv", columns=None):
    output = Path(output)
    if columns is None:
        batches = list(batches)
        columns = list(dict.fromkeys(c for b in batches for c in b.data.columns))
    columns = list(dict.fromkeys([*columns, "inference", "semantic_status", "semantic_error",
                                 "_semantic_input_file", "_semantic_input_row"]))
    totals = {"total": 0, "success": 0, "failed": 0, "pending": 0}
    first = True
    for batch in batches:
        results = [load_prediction(batch, i) for i in range(len(batch.data))]
        labeled = batch.data.copy()
        labeled["inference"] = [
            json.dumps(r["inference"], ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            if r["status"] == "ok" else "" for r in results]
        labeled["semantic_status"] = [r["status"] for r in results]
        labeled["semantic_error"] = [r["error"] for r in results]
        keep = labeled["semantic_status"].eq("ok")
        for name, frame in (("labeled", labeled), ("extracted", labeled.loc[keep])):
            (output / name).mkdir(parents=True, exist_ok=True)
            frame.to_csv(output / name / batch.source.name, index=False, encoding="utf-8")
        combined = labeled.copy()
        combined["_semantic_input_file"] = batch.source.name
        combined["_semantic_input_row"] = range(len(combined))
        if set(combined.columns) - set(columns):
            raise ValueError("Input columns changed while merging.")
        combined = combined.reindex(columns=columns)
        for name, frame in ((merged_name, combined.loc[keep]), ("unresolved_articles.csv", combined.loc[~keep])):
            frame.to_csv(output / name, mode="w" if first else "a", header=first,
                         index=False, encoding="utf-8")
        first = False
        totals["total"] += len(labeled)
        totals["success"] += int(keep.sum())
        totals["failed"] += sum(r["status"] == "failed" for r in results)
        totals["pending"] += sum(r["status"] == "pending" for r in results)
    totals["status"] = "complete" if totals["failed"] + totals["pending"] == 0 else "incomplete"
    write_json(output / "run_summary.json", totals)
    return totals


def build_parser():
    parser = StepArgumentParser(step="3_semantic_extraction", description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="relevance_filter success CSV or directory; requires TITLE and RAWTEXT.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("infer", "merge", "all"), default="all")
    parser.add_argument("--api-key", help="Runtime key; omit for hidden input. Never saved or read from the environment.")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--attempts", type=int, default=3, help="Total attempts per article, including the first.")
    parser.add_argument("--retry-delay", type=float, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--merged-name", default="extract_res.csv")
    parser.add_argument("--dry-run", action="store_true", help="Prepare snapshots/manifests without model requests.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings(api_key=args.api_key, model=args.model, base_url=args.base_url.rstrip("/"),
                        temperature=args.temperature, workers=args.workers, attempts=args.attempts,
                        timeout=args.timeout, retry_delay=args.retry_delay,
                        cache_root=getattr(args, "shared_cache_dir", None),
                        legacy_cache_roots=getattr(args, "legacy_cache_roots", ()))
    progress = None
    try:
        if (settings.workers < 1 or settings.attempts < 1 or not math.isfinite(settings.timeout)
                or settings.timeout <= 0 or not math.isfinite(settings.retry_delay) or settings.retry_delay < 0):
            raise ValueError("Invalid worker, attempt, timeout or retry-delay setting.")
        if not math.isfinite(settings.temperature) or not 0 <= settings.temperature <= 2:
            raise ValueError("Temperature must be a finite number between 0 and 2.")
        endpoint = urlsplit(settings.base_url)
        if endpoint.scheme not in {"http", "https"} or not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query:
            raise ValueError("base-url must be an HTTP(S) endpoint without embedded credentials or query parameters.")
        if not settings.model.strip():
            raise ValueError("Model name cannot be empty.")
        if Path(args.merged_name).name != args.merged_name or not args.merged_name.endswith(".csv") or args.merged_name == "unresolved_articles.csv":
            raise ValueError("merged-name must be a simple CSV filename other than unresolved_articles.csv.")
        if args.dry_run and args.stage == "merge":
            raise ValueError("dry-run prepares inputs; it cannot be combined with stage=merge.")
        if not args.dry_run and args.stage in {"infer", "all"}:
            progress = ArticleProgress("Semantic", workers=settings.workers)
        guard_step_selection(args)
        files = collect_inputs(args.input, args.output_dir)
        records = unresolved = cached_successes = 0
        plans = []
        for number, path in enumerate(files):
            if progress is not None:
                progress.preparing("Input counts", number, len(files), path.name,
                                   records=records)
            batch = prepare_batch(path, args.output_dir, settings, merge_only=args.stage == "merge",
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
                              "cached_successes": cached_successes, "requests_needed": records - cached_successes}, indent=2))
            return 0
        if args.stage in {"infer", "all"}:
            counts = _run_inference_batches(_load_batches(plans, args.output_dir, settings, merge_only=False), settings,
                                            total=records, cached_counts={},
                                            progress=progress)
            unresolved = counts["failed"]
        if args.stage in {"merge", "all"}:
            columns = list(dict.fromkeys(c for plan in plans for c in plan.columns))
            batches = _load_batches(plans, args.output_dir, settings, restore_cache=args.stage == "merge")
            summary = merge_batches(batches, args.output_dir, args.merged_name, columns)
        else:
            summary = {"stage": "infer", "status": "incomplete" if unresolved else "complete",
                       "unresolved": unresolved}
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
