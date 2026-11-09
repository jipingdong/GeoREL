"""Step 2: screen source articles for geopolitical relevance.

Input:
    A CSV or flat CSV directory containing TITLE and RAWTEXT, with source URLs,
    GDELT identifiers and other provenance columns retained when present.
Output:
    relevant_articles.csv and relevant/ feed semantic extraction. labeled/ keeps
    every normalized input row; unresolved_articles.csv retains failed/pending
    judgments. Per-batch snapshots, result caches and run_summary.json support
    reproducible merging without overwriting source files.
Execution:
    Concurrent deepseek-flash requests (temperature 0) with a parameter/hidden
    API key, bounded retries and validated success-cache reuse. infer, merge and
    all are available; merge and dry-run never call the model. False relevance
    is a completed judgment, whereas missing/failed labels remain unresolved.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from data_process.utils.batch_cache import ArticleCache, batch_directory, prepare_cached_batch, write_prediction
from data_process.utils.process_config import StepArgumentParser, guard_step_selection
from data_process.utils.source_batches import prepare_source_batches

import getpass
import hashlib
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from threading import Event
from urllib.parse import urlsplit
import pandas as pd

from data_process.utils.llm_runtime import ArticleProgress, collect_inputs, json_hash, make_client as _make_client, safe_error, write_json

SYSTEM_PROMPT = \
"""You are an expert in International Relations and Geopolitical Security analysis. Your task is to identify and filter news texts that are **relevant** to topics such as **geopolitical security**, **international events**, **inter-state relations**, **regional conflicts**, **military developments**, **political diplomacy**, **global strategy**, or **geoeconomics**.

For each news article, determine whether it is related to the aforementioned geopolitical security topics. You could make this judgment based on the following specific criteria:

1.  **Directly Relevant**:
    * **International Conflicts**: Involving military confrontations between states, territorial disputes, sanctions, threats of war, or actual acts of war.
    * **Diplomatic Events**: Significant changes in foreign policy, signing/breaking of international treaties, expulsion of diplomats, outcomes of high-level international conferences, etc.
    * **Military Developments**: Military exercises, weapon research/development/deployment, arms sales, changes in military bases, troop movements, etc.
    * **Regime Stability**: Coups, large-scale political protests, election interference, or other events that could trigger regional or international instability.
    * **Security Cooperation**: Military alliances, security agreements, counter-terrorism cooperation, joint military operations, etc.

2.  **Indirectly Relevant**:
    * **Strategic Resources**: Involving international competition for resources like energy (oil, natural gas), food, critical minerals, and water, as well as supply crises or supply chain security issues.
    * **Geoeconomics**: Economic events that significantly impact international trade patterns, investment flows, industrial chain restructuring, or involve economic sanctions and trade wars with notable effects on inter-state relations.
    * **Transnational Threats**: Cyberattacks, weapon proliferation (nuclear, biological, chemical weapons), cross-border crime (e.g., drug trafficking, terrorist financing networks), large-scale refugee movements, and other non-traditional security issues.
    * **Technological Competition**: Developments and competition in cutting-edge technologies (e.g., AI, quantum computing, space technology, biotechnology) that have a significant impact on national security and the global balance of power.
    * **Environment and Climate Change**: Large-scale natural disasters, sea-level rise due to climate change, resource scarcity, and other environmental issues that could trigger international cooperation or conflict.

3.  **Irrelevant**: The news text is invalid, or its content primarily covers areas unrelated to geopolitical security topics, such as social welfare (e.g., entertainment, sports, culture, local everyday crime, technologically insignificant advancements, general business news, etc.).

Output Format: Return only a valid JSON object with one boolean field, without explanations or Markdown.
For a relevant article, return {"relevance": true}.
For an irrelevant article, return {"relevance": false}."""

USER_PROMPT = """
The following is the news text to be analyzed:
"""


@dataclass
class Settings:
    api_key: str | None = field(default=None, repr=False)
    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
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


def normalize_source(path):
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"TITLE", "RAWTEXT"}.issubset(data.columns):
        raise ValueError(f"{Path(path).name} must contain TITLE and RAWTEXT.")
    reserved = {"relevance_status", "relevance_error", "_input_file", "_input_row"}
    if reserved.intersection(data.columns):
        raise ValueError("Use original input CSVs, not this script's labeled output.")
    data = data.drop_duplicates(subset=["TITLE", "RAWTEXT"]).reset_index(drop=True)
    data["TITLE"] = data["TITLE"].replace("", "No content")
    data["RAWTEXT"] = data["RAWTEXT"].replace("", "No content")
    return data


def prepare_batch(source, output, settings, merge_only=False, *, inspect_only=False, restore_cache=True):
    source, output = Path(source).resolve(), Path(output).resolve()
    data = normalize_source(source)
    directory = batch_directory(output, source.stem, settings)
    source_hash = hashlib.sha256(
        data.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()
    prompt_hash = json_hash([SYSTEM_PROMPT, USER_PROMPT])
    metadata = {"source_sha256": source_hash, "prompt_sha256": prompt_hash,
                "model": settings.model, "base_url": settings.base_url.rstrip("/"), "temperature": 0}
    manifest = {**metadata, "fingerprint": json_hash(metadata), "source_file": source.name, "rows": len(data)}
    batch = Batch(source, data, directory, directory / "res", manifest)
    if inspect_only:
        return batch
    return prepare_cached_batch(batch, settings, load_prediction, merge_only=merge_only,
                                stable_keys=("source_sha256", "prompt_sha256"), content=_request_content,
                                validate=lambda payload, row: parse_prediction(json.dumps(payload["inference"])),
                                restore_cache=restore_cache)


def _request_content(row):
    return USER_PROMPT + f"Title: {row['TITLE']}\nText: {row['RAWTEXT']}"


def parse_prediction(raw):
    if not isinstance(raw, str):
        raise ValueError("Model returned no text.")
    raw = raw.strip()
    fence = chr(96) * 3
    if raw.startswith(fence):
        lines = raw.splitlines()
        if len(lines) < 3 or lines[-1].strip() != fence:
            raise ValueError("Incomplete JSON code block.")
        raw = "\n".join(lines[1:-1])
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError) as error:
        raise ValueError("Model response is not a JSON object.") from error
    if not isinstance(obj, dict) or set(obj) != {"relevance"} or type(obj["relevance"]) is not bool:
        raise ValueError("Expected exactly one boolean relevance field.")
    return obj["relevance"]


def load_prediction(batch, index):
    path = batch.res_dir / f"{index}.json"
    if not path.exists():
        return {"status": "pending", "relevance": None, "error": ""}
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("fingerprint") != batch.manifest["fingerprint"] or type(item.get("idx")) is not int or item["idx"] != index:
            raise ValueError("Cached result does not match this input/configuration.")
        if item.get("status") == "ok":
            inference = item.get("inference")
            if not isinstance(inference, dict) or type(inference.get("relevance")) is not bool:
                raise ValueError("Cached relevance is not a boolean.")
            return {"status": "ok", "relevance": inference["relevance"], "error": ""}
        if item.get("status") == "failed":
            return {"status": "failed", "relevance": None, "error": str(item.get("error", "Request failed."))}
        raise ValueError("Unknown cached result status.")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        return {"status": "failed", "relevance": None, "error": f"Invalid cache: {error}"}






def _is_configuration_error(error):
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if status in {401, 402, 403, 404, 405, 422}:
        return True
    message = str(error).lower()
    return status == 400 and any(marker in message for marker in (
        "response_format", "unsupported parameter", "invalid model", "model not found",
        "model does not exist", "temperature is not supported", "temperature must",
    ))


def _extract_article(batch, index, settings, request, stop):
    if stop.is_set():
        return "pending"
    row = batch.data.iloc[index]
    content = _request_content(row)
    result = {"idx": index, "fingerprint": batch.manifest["fingerprint"]}
    for attempt in range(1, settings.attempts + 1):
        if stop.is_set():
            break
        try:
            relevance = parse_prediction(request(content))
            result.pop("error", None)
            result.update(status="ok", inference={"relevance": relevance}, attempts=attempt)
            break
        except Exception as error:
            result.update(status="failed", error=safe_error(error, settings.api_key), attempts=attempt)
            if _is_configuration_error(error):
                stop.set()
                write_prediction(batch, index, result)
                raise RuntimeError("Batch stopped after an API configuration error: " + result["error"]) from None
            if stop.is_set():
                break
            if attempt < settings.attempts:
                stop.wait(settings.retry_delay * 2 ** (attempt - 1))
    if "status" not in result:
        return "pending"
    write_prediction(batch, index, result)
    return result["status"]


def _run_inference_batches(batches, settings, request=None, *, total, cached_counts, progress=None):
    """Bound submissions across files and share one fatal-error stop signal."""
    totals = {"processed": 0, "reused": sum(cached_counts.values()), "success": 0, "failed": 0}
    owns_progress = progress is None
    if owns_progress:
        progress = ArticleProgress("Relevance", workers=settings.workers)
    stop = Event()

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
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": content}],
            temperature=0, response_format={"type": "json_object"}, stream=False,
        )
        return response.choices[0].message.content

    pending_tasks = tasks()
    try:
        progress.plan(total, totals["reused"])
        with ThreadPoolExecutor(max_workers=settings.workers) as executor:
            futures = {}
            def fill_queue():
                nonlocal client
                while not stop.is_set() and len(futures) < settings.workers * 2:
                    task = next(pending_tasks, None)
                    if task is None or stop.is_set():
                        break
                    batch, index = task
                    if request is None and client is None:
                        client = _make_client(settings)
                    future = executor.submit(_extract_article, batch, index, settings,
                                             request if request is not None else api_request, stop)
                    futures[future] = batch.source
            try:
                fill_queue()
                progress.update(totals, active=len(futures))
                while futures:
                    done, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                    for future in done:
                        status = future.result()
                        futures.pop(future)
                        # A queued worker may observe the stop before the fatal Future is handled.
                        if status == "pending":
                            continue
                        totals["processed"] += 1
                        totals["success" if status == "ok" else "failed"] += 1
                    fill_queue()
                    progress.update(totals, active=len(futures))
            except BaseException:
                stop.set()
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


def merge_batches(batches, output, merged_name="relevant_articles.csv", columns=None):
    output = Path(output)
    if columns is None:
        batches = list(batches)
        columns = list(dict.fromkeys(c for batch in batches for c in batch.data.columns))
    columns = list(dict.fromkeys([*columns, "relevance", "relevance_status", "relevance_error",
                                 "_input_file", "_input_row"]))
    totals = {"total": 0, "relevant": 0, "irrelevant": 0, "failed": 0, "pending": 0}
    first = True
    for batch in batches:
        predictions = [load_prediction(batch, i) for i in range(len(batch.data))]
        labeled = batch.data.copy()
        labeled["relevance"] = pd.array([p["relevance"] for p in predictions], dtype="boolean")
        labeled["relevance_status"] = [p["status"] for p in predictions]
        labeled["relevance_error"] = [p["error"] for p in predictions]
        (output / "labeled").mkdir(parents=True, exist_ok=True)
        labeled.to_csv(output / "labeled" / batch.source.name, index=False, encoding="utf-8")
        combined = labeled.copy()
        combined["_input_file"] = batch.source.name
        combined["_input_row"] = range(len(combined))
        keep = labeled["relevance"].fillna(False) & labeled["relevance_status"].eq("ok")
        (output / "relevant").mkdir(parents=True, exist_ok=True)
        labeled.loc[keep].to_csv(output / "relevant" / batch.source.name,
                                 index=False, encoding="utf-8")
        if set(combined.columns) - set(columns):
            raise ValueError("Input columns changed while merging.")
        combined = combined.reindex(columns=columns)
        for name, frame in (
            (merged_name, combined.loc[keep]),
            ("unresolved_articles.csv", combined.loc[~labeled["relevance_status"].eq("ok")]),
        ):
            frame.to_csv(output / name, mode="w" if first else "a", header=first,
                         index=False, encoding="utf-8")
        first = False
        totals["total"] += len(labeled)
        totals["relevant"] += int(keep.sum())
        totals["irrelevant"] += sum(p["status"] == "ok" and p["relevance"] is False for p in predictions)
        totals["failed"] += sum(p["status"] == "failed" for p in predictions)
        totals["pending"] += sum(p["status"] == "pending" for p in predictions)
    totals["status"] = "complete" if totals["failed"] + totals["pending"] == 0 else "incomplete"
    write_json(output / "run_summary.json", totals)
    return totals




def build_parser():
    parser = StepArgumentParser(step="2_relevance_filter", description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV or flat directory; requires TITLE and RAWTEXT.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("infer", "merge", "all"), default="all")
    parser.add_argument("--api-key", help="API key parameter; omit for hidden interactive input.")
    parser.add_argument("--model", default="deepseek-flash", help="Model name (default: deepseek-flash).")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--attempts", type=int, default=3, help="Maximum attempts per article, including the first.")
    parser.add_argument("--retry-delay", type=float, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--merged-name", default="relevant_articles.csv", help="Combined output filename, e.g. month_1_clean_all.csv.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare normalized snapshots/manifests without model calls.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings(api_key=args.api_key, model=args.model, base_url=args.base_url.rstrip("/"),
                        workers=args.workers, attempts=args.attempts, timeout=args.timeout,
                        retry_delay=args.retry_delay, cache_root=getattr(args, "shared_cache_dir", None),
                        legacy_cache_roots=getattr(args, "legacy_cache_roots", ()))
    progress = None
    try:
        if settings.workers < 1 or settings.attempts < 1 or settings.timeout <= 0 or settings.retry_delay < 0:
            raise ValueError("Workers/attempts must be positive; timeout > 0 and retry-delay >= 0.")
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
            progress = ArticleProgress("Relevance", workers=settings.workers)
        if args.source_batches is not None:
            source_options = dict(args.source_batches)
            if progress is not None:
                source_options["progress_fn"] = progress.preparing
            args.input = prepare_source_batches(**source_options)
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
            summary = merge_batches(batches, args.output_dir, args.merged_name, columns=columns)
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
