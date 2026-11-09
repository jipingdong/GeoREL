"""Immutable batch revisions and validated article-level result reuse."""

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable

import pandas as pd


def batch_directory(output, name, settings):
    root = settings.cache_root if settings.cache_root is not None else output
    return Path(root).expanduser().resolve() / "batches" / name


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate batch cache JSON key.")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("Nonfinite batch cache JSON number.")


def _read(path):
    value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object,
                       parse_constant=_nonfinite)
    if not isinstance(value, dict):
        raise ValueError("Batch cache metadata must be a JSON object.")
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _snapshot_matches(directory, manifest):
    try:
        snapshot = pd.read_csv(directory / "input_snapshot.csv", dtype=str, keep_default_na=False)
        digest = hashlib.sha256(snapshot.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()
        return len(snapshot) == manifest["rows"] and digest == manifest["source_sha256"]
    except (OSError, ValueError, KeyError, TypeError, UnicodeError):
        return False


def _configuration(manifest):
    return {k: v for k, v in manifest.items()
            if k not in {"source_sha256", "fingerprint", "source_file", "rows"}}


def _configuration_matches(manifest, expected, stable_keys, merge_only):
    if set(manifest) != set(expected) or type(manifest.get("rows")) is not int:
        return False
    metadata = {k: v for k, v in manifest.items() if k not in {"fingerprint", "source_file", "rows"}}
    if manifest.get("fingerprint") != _hash(metadata):
        return False
    if merge_only:
        return all(manifest.get(k) == expected.get(k) for k in stable_keys if k != "source_sha256")
    return _configuration(manifest) == _configuration(expected)


def _atomic_text(path, text, before_publish=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", delete=False,
                                         prefix=".batch-cache-", suffix=".tmp", dir=path.parent) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if before_publish is None or before_publish():
            temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _success(batch, index, load_prediction):
    try:
        result = load_prediction(batch, index)
        return result if result.get("status") == "ok" else None
    except (OSError, ValueError, TypeError, AttributeError, KeyError, IndexError):
        return None


def _payload(result):
    if "relevance" in result:
        return {"inference": {"relevance": result["relevance"]}}
    if "mapping" in result:
        return {"mapping": result["mapping"]}
    return {"inference": result["inference"]}


def _article_key(row, content):
    url = next((row[k] for k in ("url", "SOURCEURL", "source_url") if k in row), "")
    return _hash({"version": 1, "url": url, "request_content": content(row)})


@dataclass
class ArticleCache:
    directory: Path
    keys: tuple[str, ...]
    validate: Callable

    def read(self, index, row):
        try:
            item = _read(self.directory / f"{self.keys[index]}.json")
            if (item.get("version") != 1 or item.get("configuration") != self.directory.name
                    or item.get("article_key") != self.keys[index] or item.get("status") != "ok"):
                return None
            payload = _payload(item)
            if set(item) != {"version", "configuration", "article_key", "status", *payload}:
                return None
            self.validate(payload, row)
            return payload
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
            return None

    def publish(self, index, row, result):
        if result.get("status") != "ok":
            return
        payload = _payload(result)
        self.validate(payload, row)
        if self.read(index, row) is not None:
            return
        item = {"version": 1, "configuration": self.directory.name,
                "article_key": self.keys[index], "status": "ok", **payload}
        _atomic_text(self.directory / f"{self.keys[index]}.json", _json_text(item),
                     before_publish=lambda: self.read(index, row) is None)


def write_prediction(batch, index, result):
    """Checkpoint the current row and publish only its validated successful payload."""
    _atomic_text(batch.res_dir / f"{index}.json", _json_text(result))
    if batch.article_cache is not None:
        batch.article_cache.publish(index, batch.data.iloc[index], result)


def _directories(base):
    yield base
    revisions = base / "revisions"
    if revisions.is_dir():
        yield from sorted((p for p in revisions.iterdir() if p.is_dir() and not p.is_symlink()), key=str)


def _legacy_batches(batch, settings, expected, stable_keys, merge_only):
    roots = sorted({Path(root).expanduser().resolve() for root in settings.legacy_cache_roots}, key=str)
    for root in roots:
        name = batch.source.name.removesuffix(".gz").removesuffix(".csv")
        for directory in _directories(root / "batches" / name):
            if directory == batch.directory:
                continue
            try:
                manifest = _read(directory / "manifest.json")
                if (not _configuration_matches(manifest, expected, stable_keys, merge_only)
                        or not _snapshot_matches(directory, manifest)):
                    continue
                data = pd.read_csv(directory / "input_snapshot.csv", dtype=str, keep_default_na=False)
            except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
                continue
            yield replace(batch, data=data, directory=directory, res_dir=directory / "res",
                          manifest=manifest, article_cache=None)


def _import_successes(batch, candidates, load_prediction, content, validate):
    cache = batch.article_cache
    candidates = iter(candidates)
    results = [_success(batch, index, load_prediction) for index in range(len(batch.data))]
    pending = {index for index, result in enumerate(results) if result is None}
    for index, result in enumerate(results):
        if result is not None:
            cache.publish(index, batch.data.iloc[index], result)
    # Consult the article pool before opening any historical batch snapshots.
    payloads = {}
    for index in tuple(pending):
        payload = cache.read(index, batch.data.iloc[index])
        if payload is not None:
            payloads[index] = payload
            pending.remove(index)
    while pending:
        candidate = next(candidates, None)
        if candidate is None:
            break
        legacy_cache = ArticleCache(cache.directory,
                                    tuple(_article_key(row, content) for _, row in candidate.data.iterrows()), validate)
        for index in range(len(candidate.data)):
            prediction = _success(candidate, index, load_prediction)
            if prediction is not None:
                legacy_cache.publish(index, candidate.data.iloc[index], prediction)
        # The pool may now satisfy rows regardless of their former positions.
        for index in tuple(pending):
            payload = cache.read(index, batch.data.iloc[index])
            if payload is not None:
                payloads[index] = payload
                pending.remove(index)
    for index, payload in payloads.items():
        row = batch.data.iloc[index]
        result = {"idx": index, "status": "ok", "fingerprint": batch.manifest["fingerprint"], **payload}
        if "article_id" in batch.data:
            result["article_id"] = row["article_id"]
        _atomic_text(batch.res_dir / f"{index}.json", _json_text(result),
                     before_publish=lambda: _success(batch, index, load_prediction) is None)


def prepare_cached_batch(batch, settings, load_prediction, *, content, validate,
                         merge_only=False, stable_keys=(), restore_cache=True):
    """Preserve batch revisions and bind article successes to the current row order."""
    expected = batch.manifest
    base = directory = batch.directory
    pool = base.parent.parent
    source = batch.source.resolve()
    if directory == source or source.is_relative_to(directory) or directory.is_relative_to(source):
        raise ValueError("Shared batch cache must not overlap its source input.")
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        try:
            existing = _read(manifest_path)
            valid = (_configuration_matches(existing, expected, stable_keys, merge_only)
                     and _snapshot_matches(directory, existing))
        except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
            valid = False
        if not valid:
            raise ValueError(f"Batch {batch.source.name}: model/prompt/schema/aliases changed or cache is invalid; use a fresh output root.")
        if merge_only:
            metadata = {**_configuration(existing), "source_sha256": expected["source_sha256"]}
            expected = {**metadata, "fingerprint": _hash(metadata),
                        "source_file": expected["source_file"], "rows": expected["rows"]}
        if existing != expected:
            directory = base / "revisions" / expected["fingerprint"]
            manifest_path = directory / "manifest.json"
        else:
            expected = existing
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Unidentified batch cache; use a fresh output root.")
        if merge_only:
            candidate = next(_legacy_batches(batch, settings, expected, stable_keys, True), None)
            if candidate is not None:
                metadata = {**_configuration(candidate.manifest), "source_sha256": expected["source_sha256"]}
                expected = {**metadata, "fingerprint": _hash(metadata),
                            "source_file": expected["source_file"], "rows": expected["rows"]}
            elif not (pool / "articles" / ("v1-" + _hash(_configuration(expected)))).is_dir():
                raise ValueError("No matching shared or legacy batch manifest; run inference first or use a fresh output root.")
    batch = replace(batch, directory=directory, res_dir=directory / "res", manifest=expected)
    if manifest_path.exists():
        if _read(manifest_path) != expected or not _snapshot_matches(directory, expected):
            raise ValueError("Batch revision cache is invalid; use a fresh output root.")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Unidentified batch revision; use a fresh output root.")
        _atomic_text(directory / "input_snapshot.csv", batch.data.to_csv(index=False, lineterminator="\n"))
        _atomic_text(manifest_path, _json_text(batch.manifest))
    batch.res_dir.mkdir(parents=True, exist_ok=True)
    batch = replace(batch, article_cache=ArticleCache(
        pool / "articles" / ("v1-" + _hash(_configuration(batch.manifest))),
        tuple(_article_key(row, content) for _, row in batch.data.iterrows()), validate))
    roots = tuple(dict.fromkeys((pool, *settings.legacy_cache_roots)))
    candidates = _legacy_batches(batch, replace(settings, legacy_cache_roots=roots),
                                  batch.manifest, stable_keys, False)
    if restore_cache:
        _import_successes(batch, candidates, load_prediction, content, validate)
    return batch
