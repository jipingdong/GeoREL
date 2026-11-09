"""Step 1: extract article text from locally retrieved HTML.

Input:
    Timestamp folders under paths.input_dir selected by config.json batch_range.
    Each folder contains urls_title.csv and HTML referenced by html_path or index.
Output:
    Update urls_title.csv in its original folder, retaining existing columns and
    adding fulltext and extraction_status. Missing titles are filled from HTML.
Execution:
    Local extraction only; no API key or network request. Successful rows are
    skipped on rerun; failures are retried. Atomic checkpoints retain progress.
    --force explicitly re-extracts all selected rows. One batch progress bar
    reports elapsed/remaining time. Step 2 prepares TITLE/RAWTEXT working copies.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import logging
import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import repeat
from html import unescape
from urllib.parse import urlparse

from data_process.utils.process_config import StepArgumentParser, _batch_range

import pandas as pd
from bs4 import BeautifulSoup
import trafilatura
import extruct

from htmldate import find_date
import dateparser
from tqdm.auto import tqdm


TITLE_MISSING_VALUES = {"", "-1"}
ENRICH_LOCK_FILENAME = ".enrich_urls_title.running"
FULL_ENRICH_FIELDS = [
    "description",
    "publish_date",
    "publish_date_source",
    "author",
    "site_name",
    "fulltext",
    "fulltext_char_count",
    "fulltext_word_count",
    "title_source",
    "body_source",
    "html_path",
    "html_exists",
    "html_size",
    "extraction_status",
]
FIELD_PROFILES = {
    "minimal": [
        "fulltext",
        "extraction_status",
    ],
    "full": FULL_ENRICH_FIELDS,
}
DATE_META_KEYS = [
    ("property", "article:published_time"),
    ("property", "og:published_time"),
    ("property", "article:modified_time"),
    ("name", "article:published_time"),
    ("name", "publishdate"),
    ("name", "pubdate"),
    ("name", "date"),
    ("name", "dc.date"),
    ("name", "dcterms.created"),
    ("name", "dcterms.issued"),
    ("name", "parsely-pub-date"),
]
AUTHOR_META_KEYS = [
    ("name", "author"),
    ("property", "author"),
    ("name", "parsely-author"),
    ("name", "article:author"),
]
DESCRIPTION_META_KEYS = [
    ("name", "description"),
    ("property", "og:description"),
    ("name", "twitter:description"),
]
TITLE_META_KEYS = [
    ("property", "og:title"),
    ("name", "twitter:title"),
]
SITE_NAME_META_KEYS = [
    ("property", "og:site_name"),
    ("name", "application-name"),
]


def write_dataframe_csv_atomic(
    frame: pd.DataFrame,
    path: os.PathLike[str] | str,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, part_path = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".part", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(part_path, target)
    finally:
        try:
            os.remove(part_path)
        except FileNotFoundError:
            pass

def parse_month(value):
    try:
        month = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("month must be between 1 and 12") from exc
    if not 1 <= month <= 12:
        raise ValueError("month must be between 1 and 12")
    return f"{month:02d}"


def build_parser():
    parser = StepArgumentParser(step="1_enrich_urls_title", description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Crawler directory containing timestamp folders.")
    parser.add_argument("--start", required=True, help="First batch timestamp, inclusive.")
    parser.add_argument("--end", required=True, help="Last batch timestamp, inclusive.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent local batch extraction processes.")
    parser.add_argument("--write-every", type=int, default=50, help="Checkpoint every N processed rows; 0 saves at batch end.")
    parser.add_argument("--max-dirs", type=int, default=0, help="Limit selected batches for testing; 0 means all.")
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows per batch for testing; 0 means all.")
    parser.add_argument("--force", action="store_true", help="Re-extract successful rows too.")
    parser.add_argument("--no-progress", action="store_true", help="Hide the total progress bar.")
    parser.add_argument("--field-profile", choices=sorted(FIELD_PROFILES), default="minimal")
    return parser


def list_zip_dirs(base_path, year, max_dirs, input_name="urls_title.csv", month=""):
    base_dir = Path(base_path)
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    normalized_month = parse_month(month) if month else ""

    def matches_period(timestamp):
        return (not year or timestamp.startswith(year)) and (
            not normalized_month or timestamp[4:6] == normalized_month
        )

    zip_dirs = []
    for root, child_dirs, _ in os.walk(base_dir):
        directory = Path(root)
        if re.fullmatch(r"\d{14}", directory.name):
            child_dirs.clear()
            if matches_period(directory.name):
                zip_dirs.append(directory)
            if max_dirs > 0 and len(zip_dirs) >= max_dirs:
                return sorted(zip_dirs)[:max_dirs]
            continue

        child_dirs.sort()
        timestamp_names = [name for name in child_dirs if re.fullmatch(r"\d{14}", name)]
        child_dirs[:] = [name for name in child_dirs if name not in timestamp_names]
        if year:
            child_dirs[:] = [
                name
                for name in child_dirs
                if not re.fullmatch(r"\d{4}", name) or name == year
            ]

        for name in timestamp_names:
            if not matches_period(name):
                continue
            zip_dirs.append(directory / name)
            if max_dirs > 0 and len(zip_dirs) >= max_dirs:
                return sorted(zip_dirs)[:max_dirs]

    zip_dirs = sorted(zip_dirs)
    if max_dirs > 0:
        zip_dirs = zip_dirs[:max_dirs]
    return zip_dirs


def normalize_whitespace(text):
    return " ".join(stringify_value(text).split())


def stringify_value(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, (list, tuple, set)):
        parts = [stringify_value(item) for item in value]
        return " ".join(part for part in parts if part)

    if isinstance(value, dict):
        preferred_keys = (
            "name",
            "@name",
            "headline",
            "description",
            "text",
            "articleBody",
            "value",
        )
        for key in preferred_keys:
            text = stringify_value(value.get(key))
            if text:
                return text

        parts = [stringify_value(item) for item in value.values()]
        return " ".join(part for part in parts if part)

    return str(value)


def clean_text(text):
    text = unescape(stringify_value(text))
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return normalize_whitespace(text) if "\n" not in text else "\n".join(
        normalize_whitespace(line) for line in text.splitlines() if normalize_whitespace(line)
    )


def first_nonempty(*values):
    for value in values:
        if value is None:
            continue
        normalized = normalize_whitespace(value)
        if normalized and normalized not in TITLE_MISSING_VALUES:
            return normalized
    return ""


def get_domain(url):
    parsed = urlparse(url or "")
    return parsed.netloc.lower()


def get_meta_content(soup, keys):
    for attr_name, attr_value in keys:
        tag = soup.find("meta", attrs={attr_name: attr_value})
        if tag and tag.get("content"):
            return normalize_whitespace(tag["content"])
    return ""


def parse_jsonld_nodes(soup):
    nodes = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text(" ", strip=True)
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        nodes.extend(flatten_jsonld(payload))
    return nodes


def flatten_jsonld(payload):
    nodes = []
    if isinstance(payload, list):
        for item in payload:
            nodes.extend(flatten_jsonld(item))
    elif isinstance(payload, dict):
        if "@graph" in payload and isinstance(payload["@graph"], list):
            for item in payload["@graph"]:
                nodes.extend(flatten_jsonld(item))
        else:
            nodes.append(payload)
    return nodes


def normalize_author(author_value):
    if isinstance(author_value, str):
        return normalize_whitespace(author_value)
    if isinstance(author_value, dict):
        return normalize_whitespace(
            first_nonempty(
                author_value.get("name"),
                author_value.get("@name"),
            )
        )
    if isinstance(author_value, list):
        authors = [normalize_author(item) for item in author_value]
        authors = [item for item in authors if item]
        return ", ".join(dict.fromkeys(authors))
    return ""


def extract_jsonld_fields(nodes):
    data = {
        "title": "",
        "description": "",
        "publish_date": "",
        "author": "",
        "site_name": "",
        "fulltext": "",
    }

    for node in nodes:
        if not isinstance(node, dict):
            continue
        data["title"] = first_nonempty(
            data["title"],
            node.get("headline"),
            node.get("name"),
        )
        data["description"] = first_nonempty(
            data["description"],
            node.get("description"),
        )
        data["publish_date"] = first_nonempty(
            data["publish_date"],
            node.get("datePublished"),
            node.get("dateCreated"),
            node.get("dateModified"),
        )
        data["author"] = first_nonempty(
            data["author"],
            normalize_author(node.get("author")),
        )
        publisher = node.get("publisher")
        if isinstance(publisher, dict):
            data["site_name"] = first_nonempty(
                data["site_name"],
                publisher.get("name"),
            )
        data["fulltext"] = first_nonempty(
            data["fulltext"],
            node.get("articleBody"),
            node.get("text"),
        )

    return data


def extract_extruct_fields(html, base_url):
    if extruct is None:
        return {}

    try:
        payload = extruct.extract(html, base_url=base_url or "")
    except Exception:
        return {}

    nodes = []
    for key in ("json-ld", "microdata", "opengraph", "dublincore"):
        values = payload.get(key, [])
        if isinstance(values, list):
            nodes.extend(value for value in values if isinstance(value, dict))

    return extract_jsonld_fields(nodes)


def normalize_date_string(value):
    text = normalize_whitespace(value)
    if not text:
        return ""

    if dateparser is not None:
        try:
            parsed = dateparser.parse(text)
        except Exception:
            parsed = None
        if parsed is not None:
            return parsed.isoformat()

    iso_match = re.search(r"\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?", text)
    if iso_match:
        return iso_match.group(0).replace(" ", "T")

    slash_match = re.search(r"\d{4}/\d{2}/\d{2}", text)
    if slash_match:
        return slash_match.group(0).replace("/", "-")

    return text


def extract_publish_date(soup, html, jsonld_fields):
    date_value = first_nonempty(jsonld_fields.get("publish_date"))
    if date_value:
        return normalize_date_string(date_value), "jsonld"

    for attr_name, attr_value in DATE_META_KEYS:
        tag = soup.find("meta", attrs={attr_name: attr_value})
        if tag and tag.get("content"):
            normalized = normalize_date_string(tag["content"])
            if normalized:
                return normalized, f"meta:{attr_value}"

    time_tag = soup.find("time")
    if time_tag:
        normalized = normalize_date_string(
            first_nonempty(time_tag.get("datetime"), time_tag.get_text(" ", strip=True))
        )
        if normalized:
            return normalized, "time"

    if find_date is not None:
        try:
            date_value = find_date(html, extensive_search=True, original_date=True)
        except Exception:
            date_value = ""
        normalized = normalize_date_string(date_value)
        if normalized:
            return normalized, "htmldate"

    text = soup.get_text("\n", strip=True)
    heuristic = normalize_date_string(text[:5000])
    if heuristic and heuristic != text[:5000]:
        return heuristic, "heuristic"

    return "", ""


def title_needs_replacement(value):
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass

    title = normalize_whitespace(value)
    return title in TITLE_MISSING_VALUES or title.casefold().startswith("[no-title]")


def extract_title(
    soup,
    jsonld_fields,
    existing_title,
    source_url,
    trafilatura_title="",
):
    if not title_needs_replacement(existing_title):
        return normalize_whitespace(existing_title), "existing_csv"

    title = first_nonempty(trafilatura_title)
    if title:
        return title, "trafilatura"

    title = first_nonempty(
        jsonld_fields.get("title"),
        get_meta_content(soup, TITLE_META_KEYS),
        soup.title.string.strip() if soup.title and soup.title.string else "",
    )
    if title:
        return title, "metadata"

    heading = soup.find("h1")
    if heading:
        title = normalize_whitespace(heading.get_text(" ", strip=True))
        if title:
            return title, "h1"

    domain = get_domain(source_url)
    if domain:
        return f"[no-title] {domain}", "domain_fallback"
    return "[no-title]", "fallback"


def extract_description(soup, jsonld_fields):
    return first_nonempty(
        jsonld_fields.get("description"),
        get_meta_content(soup, DESCRIPTION_META_KEYS),
    )


def extract_author(soup, jsonld_fields):
    author = first_nonempty(
        jsonld_fields.get("author"),
        get_meta_content(soup, AUTHOR_META_KEYS),
    )
    return normalize_whitespace(author)


def extract_site_name(soup, jsonld_fields, source_url):
    return first_nonempty(
        jsonld_fields.get("site_name"),
        get_meta_content(soup, SITE_NAME_META_KEYS),
        get_domain(source_url),
    )


def extract_trafilatura_fields(html, source_url=""):
    if trafilatura is None:
        return "", "", ""

    try:
        document = trafilatura.bare_extraction(
            html,
            url=source_url or None,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_precision=True,
            with_metadata=True,
        )
    except Exception:
        return "", "", ""

    if document is None:
        return "", "", ""

    if isinstance(document, dict):
        title = document.get("title", "")
        text = document.get("text", "") or document.get("raw_text", "")
    else:
        title = getattr(document, "title", "")
        text = getattr(document, "text", "") or getattr(document, "raw_text", "")

    cleaned_title = normalize_whitespace(title)
    cleaned = clean_text(text or "")
    if cleaned:
        return cleaned_title, cleaned, "trafilatura"
    return cleaned_title, "", ""


def extract_fulltext_trafilatura(html, source_url=""):
    _, fulltext, body_source = extract_trafilatura_fields(html, source_url)
    return fulltext, body_source


def extract_fulltext_fallback(soup, jsonld_fields):
    article_body = clean_text(jsonld_fields.get("fulltext", ""))
    if article_body:
        return article_body, "jsonld"

    working_soup = BeautifulSoup(str(soup), "html.parser")
    for tag in working_soup(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "form",
            "header",
            "footer",
            "nav",
            "aside",
            "iframe",
        ]
    ):
        tag.decompose()

    content_root = (
        working_soup.find("article")
        or working_soup.find("main")
        or working_soup.find(attrs={"role": "main"})
        or working_soup.find(attrs={"itemprop": "articleBody"})
        or working_soup.body
        or working_soup
    )

    blocks = []
    seen = set()
    for tag in content_root.find_all(["h1", "h2", "h3", "p", "blockquote", "li"]):
        text = clean_text(tag.get_text(" ", strip=True))
        if len(text) < 30 or text in seen:
            continue
        seen.add(text)
        blocks.append(text)

    if not blocks:
        text = content_root.get_text("\n", strip=True)
        blocks = [clean_text(line) for line in text.splitlines() if len(clean_text(line)) >= 30]

    fulltext = "\n".join(blocks).strip()
    if fulltext:
        return fulltext, "bs4"
    return "", ""


def get_active_fields(field_profile):
    return FIELD_PROFILES[field_profile]


def ensure_columns(df, active_fields):
    if "title" not in df.columns:
        df["title"] = ""

    for field in active_fields:
        if field not in df.columns:
            if field in {"fulltext_char_count", "fulltext_word_count", "html_size"}:
                df[field] = 0
            elif field == "html_exists":
                df[field] = False
            else:
                df[field] = ""
    return df


def row_needs_processing(row, force):
    if force:
        return True

    existing_fulltext = normalize_whitespace(str(row.get("fulltext", "")))
    if (existing_fulltext and row.get("extraction_status") == "ok"
            and not title_needs_replacement(row.get("title", ""))):
        return False

    return True


def normalize_html_relative_path(value):
    relative_path = normalize_whitespace(value).replace("\\", "/")
    if not relative_path:
        return ""

    path = Path(relative_path)
    if path.is_absolute() or path.drive or ".." in path.parts:
        return ""
    return relative_path.lstrip("/")


def resolve_html_path(row, zip_dir):
    zip_dir = Path(zip_dir)
    declared_relative = normalize_html_relative_path(row.get("html_path", ""))
    candidates = []
    if declared_relative:
        candidates.append((declared_relative, zip_dir.joinpath(*declared_relative.split("/"))))

    idx = normalize_whitespace(row.get("index", ""))
    if idx:
        legacy_relative = f"htmls/{idx}.html"
        if legacy_relative != declared_relative:
            candidates.append((legacy_relative, zip_dir / "htmls" / f"{idx}.html"))

    for relative_path, absolute_path in candidates:
        if absolute_path.is_file():
            return relative_path, absolute_path

    if candidates:
        return candidates[0]
    return "", zip_dir / "htmls" / ".missing-index.html"


def enrich_row(row, zip_dir, field_profile):
    active_fields = set(get_active_fields(field_profile))
    source_url = str(row.get("SOURCEURL", "") or "")
    html_rel_path, html_path = resolve_html_path(row, zip_dir)

    result = {}
    if "html_path" in active_fields:
        result["html_path"] = html_rel_path
    if "html_exists" in active_fields:
        result["html_exists"] = html_path.exists()
    if "html_size" in active_fields:
        result["html_size"] = html_path.stat().st_size if html_path.exists() else 0
    if "extraction_status" in active_fields:
        result["extraction_status"] = ""
    if "description" in active_fields:
        result["description"] = ""
    if "publish_date" in active_fields:
        result["publish_date"] = ""
    if "publish_date_source" in active_fields:
        result["publish_date_source"] = ""
    if "author" in active_fields:
        result["author"] = ""
    if "site_name" in active_fields:
        result["site_name"] = get_domain(source_url)
    if "fulltext" in active_fields:
        result["fulltext"] = ""
    if "fulltext_char_count" in active_fields:
        result["fulltext_char_count"] = 0
    if "fulltext_word_count" in active_fields:
        result["fulltext_word_count"] = 0
    if "title_source" in active_fields:
        result["title_source"] = ""
    if "body_source" in active_fields:
        result["body_source"] = ""

    if not html_path.exists():
        if "extraction_status" in active_fields:
            result["extraction_status"] = "missing_html"
        return result

    if html_path.stat().st_size == 0:
        if "extraction_status" in active_fields:
            result["extraction_status"] = "empty_html"
        return result

    try:
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        if "extraction_status" in active_fields:
            result["extraction_status"] = "parse_error"
        return result

    jsonld_nodes = parse_jsonld_nodes(soup)
    jsonld_fields = extract_jsonld_fields(jsonld_nodes)
    if field_profile == "full" and extruct is not None:
        extruct_fields = extract_extruct_fields(html, source_url)
        for key, value in extruct_fields.items():
            jsonld_fields[key] = first_nonempty(jsonld_fields.get(key, ""), value)

    trafilatura_title, fulltext, body_source = extract_trafilatura_fields(html, source_url)
    title, title_source = extract_title(
        soup,
        jsonld_fields,
        row.get("title", ""),
        source_url,
        trafilatura_title,
    )
    if not fulltext:
        fulltext, body_source = extract_fulltext_fallback(soup, jsonld_fields)

    fulltext = fulltext.strip()
    result["title"] = title
    if "title_source" in active_fields:
        result["title_source"] = title_source
    if "description" in active_fields:
        result["description"] = extract_description(soup, jsonld_fields)
    if "publish_date" in active_fields or "publish_date_source" in active_fields:
        publish_date, publish_date_source = extract_publish_date(soup, html, jsonld_fields)
        if "publish_date" in active_fields:
            result["publish_date"] = publish_date
        if "publish_date_source" in active_fields:
            result["publish_date_source"] = publish_date_source
    if "author" in active_fields:
        result["author"] = extract_author(soup, jsonld_fields)
    if "site_name" in active_fields:
        result["site_name"] = extract_site_name(soup, jsonld_fields, source_url)
    if "fulltext" in active_fields:
        result["fulltext"] = fulltext
    if "fulltext_char_count" in active_fields:
        result["fulltext_char_count"] = len(fulltext)
    if "fulltext_word_count" in active_fields:
        result["fulltext_word_count"] = len(fulltext.split()) if fulltext else 0
    if "body_source" in active_fields:
        result["body_source"] = body_source
    if "extraction_status" in active_fields:
        result["extraction_status"] = "ok" if fulltext else "empty_extraction"
    return result


def build_row_tasks(df, max_rows, force):
    row_indices = list(df.index)
    if max_rows > 0:
        row_indices = row_indices[:max_rows]

    tasks = []
    for row_idx in row_indices:
        row = df.loc[row_idx]
        if not row_needs_processing(row, force):
            continue
        tasks.append((row_idx, row.to_dict()))
    return tasks


def apply_enriched_result(df, row_idx, enriched):
    for key, value in enriched.items():
        df.loc[row_idx, key] = value

    if enriched.get("title"):
        df.loc[row_idx, "title"] = enriched["title"]
    return df


def process_row_task(task, zip_dir, field_profile):
    row_idx, row_data = task
    enriched = enrich_row(row_data, zip_dir, field_profile)
    return row_idx, enriched


def process_zip_dir_serial(
    df,
    zip_dir,
    output_path,
    tasks,
    write_every,
    zip_name,
    field_profile,
    show_progress,
):
    processed = 0
    updated = 0

    progress_bar = tqdm(
        total=len(tasks),
        desc=f"{zip_name} rows",
        unit="row",
        leave=False,
        disable=(not show_progress or len(tasks) == 0),
    )
    for row_idx, row_data in tasks:
        enriched = enrich_row(row_data, zip_dir, field_profile)
        df = apply_enriched_result(df, row_idx, enriched)

        processed += 1
        if enriched.get("extraction_status") == "ok":
            updated += 1
        progress_bar.update(1)

        if write_every > 0 and processed % write_every == 0:
            write_dataframe_csv_atomic(df, output_path)
            logging.debug(
                "%s: checkpoint saved after %s processed rows",
                zip_name,
                processed,
            )

    progress_bar.close()
    return df, processed, updated


def process_zip_dir_rows_parallel(
    df,
    zip_dir,
    output_path,
    tasks,
    write_every,
    zip_name,
    row_workers,
    field_profile,
    show_progress,
):
    processed = 0
    updated = 0
    futures = {}

    progress_bar = tqdm(
        total=len(tasks),
        desc=f"{zip_name} rows",
        unit="row",
        leave=False,
        disable=(not show_progress or len(tasks) == 0),
    )
    with ThreadPoolExecutor(max_workers=max(1, row_workers)) as executor:
        for task in tasks:
            future = executor.submit(process_row_task, task, zip_dir, field_profile)
            futures[future] = task[0]

        for future in as_completed(futures):
            row_idx = futures[future]
            try:
                _, enriched = future.result()
            except Exception as exc:
                logging.exception("%s: row worker failed for index %s: %s", zip_name, row_idx, exc)
                continue

            df = apply_enriched_result(df, row_idx, enriched)
            processed += 1
            if enriched.get("extraction_status") == "ok":
                updated += 1
            progress_bar.update(1)

            if write_every > 0 and processed % write_every == 0:
                write_dataframe_csv_atomic(df, output_path)
                logging.debug(
                    "%s: checkpoint saved after %s processed rows",
                    zip_name,
                    processed,
                )

    progress_bar.close()
    return df, processed, updated


def process_zip_dir(
    zip_dir,
    input_name,
    output_name,
    max_rows,
    write_every,
    force,
    field_profile="minimal",
    parallel_mode="none",
    row_workers=8,
    show_progress=True,
):
    csv_path = zip_dir / input_name
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        raise ValueError(f"{zip_dir.name}: {input_name} is missing or empty")

    output_path = zip_dir / output_name
    lock_path = zip_dir / ENRICH_LOCK_FILENAME

    with lock_path.open("x", encoding="utf-8") as handle:
        handle.write("running\n")
    try:
        df = pd.read_csv(csv_path, keep_default_na=False, dtype=str)
        df = ensure_columns(df, get_active_fields(field_profile))
        tasks = build_row_tasks(df, max_rows, force)

        if parallel_mode == "rows":
            df, processed, updated = process_zip_dir_rows_parallel(
                df=df,
                zip_dir=zip_dir,
                output_path=output_path,
                tasks=tasks,
                write_every=write_every,
                zip_name=zip_dir.name,
                row_workers=row_workers,
                field_profile=field_profile,
                show_progress=show_progress,
            )
        else:
            df, processed, updated = process_zip_dir_serial(
                df=df,
                zip_dir=zip_dir,
                output_path=output_path,
                tasks=tasks,
                write_every=write_every,
                zip_name=zip_dir.name,
                field_profile=field_profile,
                show_progress=show_progress,
            )

        write_dataframe_csv_atomic(df, output_path)
        logging.debug(
            "%s: final save completed, processed=%s, ok=%s, output=%s",
            zip_dir.name,
            processed,
            updated,
            output_path,
        )
        return {
            "zip_name": zip_dir.name,
            "processed_rows": processed,
            "ok_rows": updated,
            "failed_rows": int((df["extraction_status"] != "ok").sum()),
            "output_path": str(output_path),
        }
    finally:
        if lock_path.exists():
            lock_path.unlink()


def process_zip_dir_safe(
    zip_dir,
    input_name,
    output_name,
    max_rows,
    write_every,
    force,
    field_profile,
    parallel_mode,
    row_workers,
    show_progress,
):
    try:
        summary = process_zip_dir(
            zip_dir=zip_dir,
            input_name=input_name,
            output_name=output_name,
            max_rows=max_rows,
            write_every=write_every,
            force=force,
            field_profile=field_profile,
            parallel_mode=parallel_mode,
            row_workers=row_workers,
            show_progress=show_progress,
        )
        return {
            "zip_name": zip_dir.name,
            "summary": summary,
            "error": "",
        }
    except Exception as exc:
        return {
            "zip_name": zip_dir.name,
            "summary": None,
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        _batch_range({"start": args.start, "end": args.end})
        if args.workers < 1 or min(args.max_dirs, args.max_rows, args.write_every) < 0:
            raise ValueError("workers must be positive; row, batch and checkpoint limits must be nonnegative.")
        zip_dirs = [folder for folder in list_zip_dirs(args.input, "", 0)
                    if args.start <= folder.name <= args.end]
        if args.max_dirs:
            zip_dirs = zip_dirs[:args.max_dirs]
        if not zip_dirs:
            raise ValueError("No timestamp folders in the selected batch range.")
        summaries, errors = [], []
        with tqdm(total=len(zip_dirs), desc="Extracting HTML", unit="batch", ascii=True,
                  dynamic_ncols=True, disable=args.no_progress) as progress:
            if args.workers == 1:
                results = map(
                    process_zip_dir_safe, zip_dirs, repeat("urls_title.csv"), repeat("urls_title.csv"),
                    repeat(args.max_rows), repeat(args.write_every), repeat(args.force),
                    repeat(args.field_profile), repeat("none"), repeat(1), repeat(False))
                for result in results:
                    if result["error"]:
                        errors.append(result)
                        tqdm.write(f"{result['zip_name']}: {result['error']}", file=sys.stderr)
                    else:
                        summaries.append(result["summary"])
                    progress.update(1)
            else:
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    futures = [executor.submit(
                        process_zip_dir_safe, folder, "urls_title.csv", "urls_title.csv",
                        args.max_rows, args.write_every, args.force, args.field_profile,
                        "none", 1, False) for folder in zip_dirs]
                    for future in as_completed(futures):
                        result = future.result()
                        if result["error"]:
                            errors.append(result)
                            tqdm.write(f"{result['zip_name']}: {result['error']}", file=sys.stderr)
                        else:
                            summaries.append(result["summary"])
                        progress.update(1)
        failed_rows = sum(item["failed_rows"] for item in summaries)
        print(json.dumps({
            "status": "incomplete" if errors or failed_rows else "complete",
            "batches": len(zip_dirs), "failed_batches": len(errors),
            "processed_rows": sum(item["processed_rows"] for item in summaries),
            "ok_rows": sum(item["ok_rows"] for item in summaries),
            "failed_rows": failed_rows,
        }, ensure_ascii=False, indent=2))
        return 2 if errors or failed_rows else 0
    except (OSError, ValueError) as error:
        print(f"HTML extraction failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
