"""Shared Runtime: helpers used by model-based steps; no independent pipeline step.

Input:
    Stage settings, explicit file paths and JSON-serializable values supplied
    by relevance, semantic, entity, time/location and country-processing modules.
Output:
    Configured API clients, stable content hashes, redacted error text and JSON
    files. These helpers do not extract fields or generate a dataset on their own.
Execution:
    No work or model request runs at import time. Client creation accepts a
    runtime API key or hidden interactive input, never an environment lookup;
    SDK-internal retries are disabled so each stage controls its own attempt limit.
"""

import getpass
import hashlib
import json
from pathlib import Path

from tqdm import tqdm


class ArticleProgress:
    """Render one shared bar for pending article requests across all batches."""

    def __init__(self, desc, *, workers, total=None, cached=0):
        self.desc = desc
        self.workers = workers
        self.total = None
        self.closed = False
        self.preparation_phase = None
        self.bar = tqdm(
            total=None, desc="Preparing", unit="article", ascii=True,
            dynamic_ncols=True, disable=False,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}{postfix}]",
        )
        if total is not None:
            self.plan(total, cached)

    def preparing(self, phase, completed=0, total=None, batch="", *, records=None, cached=None):
        """Expose local file work before the pending-article total is known."""
        if self.closed:
            return
        self.bar.set_description_str("Preparing: " + phase, refresh=False)
        if self.preparation_phase != phase or self.bar.total != total:
            self.bar.unit = "file"
            self.bar.reset(total=total)
            self.preparation_phase = phase
        self.bar.n = completed
        counters = {}
        if batch:
            counters["file"] = batch
        if records is not None:
            counters["articles"] = records
        if cached is not None:
            counters["cached"] = cached
        self.bar.set_postfix(counters)

    def plan(self, total, cached):
        self.total = total
        self.bar.unit = "article"
        self.bar.set_description_str(self.desc, refresh=False)
        self.bar.set_postfix(cached=cached, failed=0, workers=self.workers, refresh=False)
        self.bar.reset(total=max(0, total - cached))

    def update(self, totals, active=0):
        if self.closed:
            return
        self.bar.total = max(0, self.total - totals["reused"])
        self.bar.set_postfix(
            cached=totals["reused"], failed=totals["failed"],
            workers=self.workers, refresh=False,
        )
        delta = totals["processed"] - self.bar.n
        if delta:
            self.bar.update(delta)
        else:
            self.bar.refresh()

    def close(self):
        if not self.closed:
            self.closed = True
            self.bar.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def json_hash(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def safe_error(error, api_key):
    text = str(error)
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    return text[:1000]


def make_client(settings):
    from openai import OpenAI
    if not settings.api_key:
        settings.api_key = getpass.getpass("API key (hidden input): ").strip()
    if not settings.api_key:
        raise ValueError("An API key is required for inference.")
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url,
                  timeout=settings.timeout, max_retries=0)


def collect_inputs(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Input and output paths must be separate, non-overlapping locations.")
    files = [source] if source.is_file() else list(source.glob("*.csv")) if source.is_dir() else []
    if not files or any(p.suffix.lower() != ".csv" for p in files):
        raise ValueError("Input must be a CSV file or a directory containing CSV files.")
    return sorted(files, key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.name.lower()))
