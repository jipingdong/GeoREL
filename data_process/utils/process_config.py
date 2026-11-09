"""Read-only configuration parsing and runtime selection claims for each step."""

import argparse
from datetime import datetime
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.json"
STEP_NAMES = (
    "enrich_urls_title", "relevance_filter", "semantic_extraction",
    "spatiotemporal_extraction", "geocode_locations", "normalize_countries",
    "finalize_dataset",
)
STEP_KEYS = tuple(f"{i}_{name}" for i, name in enumerate(STEP_NAMES, 1))
MODEL_STEPS = {1, 2, 3, 5}
LLM_KEYS = {"api_key", "model", "base_url", "workers", "attempts", "timeout", "retry_delay"}
PATH_KEYS = {"input", "output_dir", "output", "cache", "aliases", "report_dir"}
SOURCE_DEFAULTS = {
    "title_column": "title", "text_column": "fulltext",
    "status_column": "extraction_status", "success_status": "ok",
}
OUTPUT_DIRS = (None, "02_relevance", "03_semantic", "04_spacetime", "05_geocoded", "06_countries")
INPUT_SUFFIXES = (None, "relevant", "extracted", "extracted",
                  "geocoded_articles.csv", "normalized_articles.csv")


class _ConfigError(ValueError):
    pass


def _object(value):
    if not isinstance(value, dict):
        raise _ConfigError("Configuration sections must be JSON objects.")
    return value


def _check_keys(value, allowed):
    if set(value) - set(allowed):
        raise _ConfigError("Unknown configuration field.")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _ConfigError("Duplicate JSON configuration field.")
        result[key] = value
    return result


def _invalid_constant(value):
    raise _ConfigError("Configuration numbers must be finite.")


def _read_config(path):
    try:
        return _object(json.loads(path.read_text(encoding="utf-8-sig"),
                                  object_pairs_hook=_unique_object,
                                  parse_constant=_invalid_constant))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        raise _ConfigError("Cannot read a valid JSON processing configuration.") from None


def _path(value, base):
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise _ConfigError("Configured paths must be nonempty strings.")
    try:
        path = Path(value).expanduser()
        return (base / path).resolve() if not path.is_absolute() else path.resolve()
    except (OSError, ValueError, RuntimeError):
        raise _ConfigError("Invalid configured path.") from None


def _actions(parser):
    result = {}
    for action in parser._actions:
        if action.dest not in {"help", "config", "no_config"}:
            result.setdefault(action.dest, action)
    return result


def _value(value, action, base):
    if value is None:
        if action.default is None and not action.required:
            return None
        raise _ConfigError("Null is not valid for this configuration option.")
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        valid = isinstance(value, bool)
    elif action.type is int:
        valid = type(value) is int
    elif action.type is float:
        valid = type(value) in {int, float}
    else:
        valid = isinstance(value, str)
    if not valid:
        raise _ConfigError("Configuration option has an invalid JSON type.")
    if type(value) in {int, float}:
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise _ConfigError("Configuration numbers must be finite.")
    if action.choices is not None and value not in action.choices:
        raise _ConfigError("Configuration option is not an allowed choice.")
    if action.dest in PATH_KEYS or action.type is Path:
        path = _path(value, base)
        return path if action.type is Path else str(path)
    if action.type is float:
        return float(value)
    return value


def _batch_range(value):
    value = _object(value)
    _check_keys(value, {"start", "end"})
    if set(value) != {"start", "end"}:
        raise _ConfigError("batch_range requires start and end.")
    for stamp in value.values():
        if not isinstance(stamp, str) or re.fullmatch(r"[0-9]{14}", stamp) is None:
            raise _ConfigError("Batch bounds must be 14-digit timestamp strings.")
        try:
            datetime.strptime(stamp, "%Y%m%d%H%M%S")
        except ValueError:
            raise _ConfigError("Batch bounds must be valid calendar timestamps.") from None
    if value["end"] < value["start"]:
        raise _ConfigError("Batch end must not precede batch start.")
    return value


def _protect_source_outputs(source, arguments):
    for name in ("output_dir", "output", "cache", "report_dir", "shared_cache_dir"):
        value = getattr(arguments, name, None)
        if value is None:
            continue
        target = Path(value).expanduser().resolve()
        if target == source or target.is_relative_to(source) or source.is_relative_to(target):
            raise _ConfigError("Output, cache and report paths must not overlap the configured source input.")


def _cache_locations(arguments, step, configuration, base):
    arguments.shared_cache_dir = None
    arguments.legacy_cache_roots = ()
    index = STEP_KEYS.index(step)
    if configuration is None or index == 0 or index >= len(OUTPUT_DIRS):
        return
    output_root = configuration.get("paths", {}).get("output_dir")
    if output_root is None:
        return
    workspace = _path(output_root, base)
    shared = workspace / ".cache" / OUTPUT_DIRS[index]
    if index == 4 and arguments.cache is not None:
        shared = Path(arguments.cache).expanduser().resolve()
    arguments.shared_cache_dir = shared
    candidates = [Path(arguments.output_dir).expanduser().resolve()]
    if workspace.is_dir():
        for folder in workspace.iterdir():
            if (re.fullmatch(r"[0-9]{14}_[0-9]{14}", folder.name)
                    and folder.is_dir() and not folder.is_symlink()):
                candidates.append(folder / OUTPUT_DIRS[index])
    arguments.legacy_cache_roots = tuple(sorted(set(candidates), key=lambda path: str(path).lower()))


def _check_selection_marker(marker, identity):
    try:
        existing = _read_config(marker)
    except _ConfigError:
        raise ValueError("Invalid selection marker; use a fresh output target.") from None
    if existing != identity:
        raise ValueError("Output selection changed; use a fresh output target.")


def _check_upstream_selection(required, identity):
    message = ("Upstream selection is missing or changed; rerun the previous step with config "
               "or provide an explicit input.")
    try:
        existing = _read_config(Path(required["marker"]))
    except _ConfigError:
        raise ValueError(message) from None
    expected = {key: identity[key] for key in ("version", "raw_source", "start", "end")}
    expected.update(step=required["step"], output=required["output"])
    if any(existing.get(key) != value for key, value in expected.items()):
        raise ValueError(message)
    summary_path = Path(required["output"]) / "run_summary.json"
    if summary_path.exists():
        try:
            summary = _read_config(summary_path)
        except _ConfigError:
            raise ValueError(message) from None
        if summary.get("status") == "stale":
            raise ValueError("Upstream aggregate is stale; run the previous step with stage=merge or all.")


def guard_step_selection(args):
    """Claim an output for one configured selection, without invoking processing."""
    metadata = getattr(args, "selection_guard", None)
    if metadata is None:
        return
    source = Path(args.input).expanduser().resolve()
    if not source.exists():
        raise ValueError("Input must exist before claiming a step output.")
    target = Path(getattr(args, "output_dir", None) or args.output).expanduser().resolve()
    identity = {key: value for key, value in metadata.items() if key != "upstream"}
    identity.update(input=str(source), output=str(target))
    if "upstream" in metadata:
        _check_upstream_selection(metadata["upstream"], identity)
    marker = Path(str(target) + ".selection.json")
    if marker.exists():
        _check_selection_marker(marker, identity)
        return
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        if marker.exists():
            _check_selection_marker(marker, identity)
            return
        raise ValueError("Unclaimed existing output; use a fresh output target for this selection.")

    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         prefix=".selection-", suffix=".tmp", dir=marker.parent,
                                         delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(identity, handle, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Publishing a fully written hard link is atomic and never replaces a winner.
        try:
            os.link(temporary, marker)
        except FileExistsError:
            _check_selection_marker(marker, identity)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class StepArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, step="1_enrich_urls_title", **kwargs):
        if step not in STEP_KEYS:
            raise ValueError("Unsupported processing step.")
        kwargs["allow_abbrev"] = False
        super().__init__(*args, **kwargs)
        self.step = step
        group = self.add_mutually_exclusive_group()
        group.add_argument("--config", type=Path, help="Read this JSON config; relative paths use its directory.")
        group.add_argument("--no-config", action="store_true", help="Ignore the adjacent default config.json.")

    def _negative_flags(self):
        for action in list(self._actions):
            if isinstance(action, argparse._StoreTrueAction) and action.dest != "no_config":
                option = next((x for x in action.option_strings if x.startswith("--")), None)
                negative = "--no-" + option[2:] if option else None
                if negative and negative not in self._option_string_actions:
                    self.add_argument(negative, dest=action.dest, action="store_false",
                                      default=argparse.SUPPRESS, help="Override the configured boolean flag.")

    def _step_parsers(self):
        parsers = {}
        for key, name in zip(STEP_KEYS, STEP_NAMES):
            if key == self.step:
                parsers[key] = self
            else:
                module = importlib.import_module(f"data_process.{name}")
                parsers[key] = module.build_parser()
        return parsers

    def _configured_defaults(self, config, base, explicit_input):
        _check_keys(config, {"paths", "batch_range", "llm", "steps"})
        paths = _object(config.get("paths", {}))
        _check_keys(paths, {"input_dir", "output_dir"})
        paths = {key: _path(value, base) for key, value in paths.items()}
        bounds = _batch_range(config["batch_range"]) if "batch_range" in config else None
        shared = _object(config.get("llm", {}))
        _check_keys(shared, LLM_KEYS)
        steps = _object(config.get("steps", {}))
        _check_keys(steps, STEP_KEYS)
        parsers = self._step_parsers()
        actions = {key: _actions(parser) for key, parser in parsers.items()}
        shared = {key: _value(value, actions[STEP_KEYS[1]][key], base) for key, value in shared.items()}

        # Validate every section, even when this invocation runs a different step.
        settings = {}
        for key in STEP_KEYS:
            options = _object(steps.get(key, {}))
            extra = SOURCE_DEFAULTS if key == STEP_KEYS[1] else {}
            _check_keys(options, set(actions[key]) | set(extra))
            settings[key] = {}
            for name, value in options.items():
                if name in extra:
                    if not isinstance(value, str) or not value.strip():
                        raise _ConfigError("Source column/status settings must be nonempty strings.")
                    settings[key][name] = value
                else:
                    settings[key][name] = _value(value, actions[key][name], base)

        scope = paths["output_dir"] / f"{bounds['start']}_{bounds['end']}" if bounds and "output_dir" in paths else None
        directories = [Path(settings[key]["output_dir"]) if "output_dir" in settings[key]
                       else scope / name if scope and name else None
                       for key, name in zip(STEP_KEYS[:6], OUTPUT_DIRS)]
        index = STEP_KEYS.index(self.step)
        own = settings[self.step]
        upstream_index = index - 1
        defaults = {key: value for key, value in shared.items()
                    if index in MODEL_STEPS and key in actions[self.step]}
        if index < 6 and directories[index] is not None:
            defaults["output_dir"] = _value(str(directories[index]), actions[self.step]["output_dir"], base)
        elif index == 6 and scope is not None:
            defaults["output"] = _value(str(scope / "07_release/georel_dataset.csv"), actions[self.step]["output"], base)
        source = paths.get("input_dir") if index == 0 else scope / "00_input/articles" if index == 1 and scope else None
        if index >= 2 and directories[upstream_index] is not None:
            source = directories[upstream_index] / INPUT_SUFFIXES[upstream_index]
        if source is not None:
            defaults["input"] = _value(str(source), actions[self.step]["input"], base)
        if index == 0 and bounds:
            defaults.update(start=bounds["start"], end=bounds["end"])
        defaults.update({key: value for key, value in own.items() if key not in SOURCE_DEFAULTS})
        if index == 4 and "output_dir" in paths and defaults.get("cache") is None:
            defaults["cache"] = str(paths["output_dir"] / ".cache" / OUTPUT_DIRS[index])

        source_batches = None
        if index == 1 and "input_dir" in paths and not explicit_input and "input" not in own:
            if scope is None:
                raise _ConfigError("Source batches require paths.output_dir and batch_range.")
            source_batches = {"input_dir": paths["input_dir"], "prepared_dir": scope / "00_input",
                              "start": bounds["start"], "end": bounds["end"],
                              **{key: own.get(key, default) for key, default in SOURCE_DEFAULTS.items()}}
        upstream = None
        if index >= 2 and source is not None and not explicit_input and "input" not in own:
            previous_output = str(directories[upstream_index])
            upstream = {"marker": previous_output + ".selection.json",
                        "step": STEP_KEYS[upstream_index], "output": previous_output}
        return defaults, source_batches, upstream

    def parse_args(self, args=None, namespace=None):
        arguments = list(sys.argv[1:] if args is None else args)
        self._negative_flags()
        if any(token in {"-h", "--help"} for token in arguments):
            return super().parse_args(arguments, namespace)

        selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        group = selector.add_mutually_exclusive_group()
        group.add_argument("--config", type=Path)
        group.add_argument("--no-config", action="store_true")
        selection, _ = selector.parse_known_args(arguments)
        input_options = set(_actions(self).get("input").option_strings) if "input" in _actions(self) else set()
        explicit_input = any(token.split("=", 1)[0] in input_options for token in arguments)
        config_path = selection.config
        if config_path is None and not selection.no_config and not explicit_input and DEFAULT_CONFIG.is_file():
            config_path = DEFAULT_CONFIG

        defaults, source_batches, protected_source, upstream = {}, None, None, None
        if config_path is not None:
            try:
                config_path = config_path.expanduser().resolve()
                configuration = _read_config(config_path)
                defaults, source_batches, upstream = self._configured_defaults(
                    configuration, config_path.parent, explicit_input)
                if "input_dir" in configuration.get("paths", {}):
                    protected_source = _path(configuration["paths"]["input_dir"], config_path.parent)
            except _ConfigError as error:
                self.error(str(error))
            except (OSError, ValueError, RuntimeError):
                self.error("Invalid processing configuration.")
            defaults["config"] = config_path

        # argparse defaults and required flags belong to the parser, not a single call.
        saved = [(action, action.default, action.required) for action in self._actions]
        original_defaults = self._defaults.copy()
        try:
            self.set_defaults(**defaults)
            for action in self._actions:
                if action.dest in defaults:
                    action.required = False
            result = super().parse_args(arguments, namespace)
            try:
                _cache_locations(result, self.step, configuration if config_path is not None else None,
                                 config_path.parent if config_path is not None else None)
            except _ConfigError as error:
                self.error(str(error))
            except (OSError, ValueError, RuntimeError):
                self.error("Invalid configured cache location.")
            if protected_source is not None and self.step != "1_enrich_urls_title":
                try:
                    _protect_source_outputs(protected_source, result)
                except _ConfigError as error:
                    self.error(str(error))
                except (OSError, ValueError, RuntimeError):
                    self.error("Invalid configured output path.")
            result.source_batches = source_batches
            result.selection_guard = None
            if protected_source is not None and "batch_range" in configuration and self.step != "1_enrich_urls_title":
                try:
                    output = getattr(result, "output_dir", None) or result.output
                    result.selection_guard = {
                        "version": 1, "step": self.step, "raw_source": str(protected_source),
                        "start": configuration["batch_range"]["start"],
                        "end": configuration["batch_range"]["end"],
                        "input": str(Path(result.input).expanduser().resolve()),
                        "output": str(Path(output).expanduser().resolve()),
                    }
                    if upstream is not None:
                        result.selection_guard["upstream"] = upstream
                except (OSError, ValueError, RuntimeError):
                    self.error("Invalid configured selection path.")
            return result
        finally:
            self._defaults = original_defaults
            for action, default, required in saved:
                action.default, action.required = default, required
