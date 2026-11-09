"""Atomic ZIP, CSV-title, and hash-addressed HTML storage."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup

from data_crawl.utils.paths import absolute_html_path, html_relative_path_for_url


TITLE_MISSING_VALUES = ("", "-1")
REQUIRED_EVENT_COLUMNS = [
    "GLOBALEVENTID",
    "Actor1CountryCode",
    "Actor2CountryCode",
    "SOURCEURL",
]
EVENT_DTYPES = {
    "GLOBALEVENTID": "Int64",
    "Actor1CountryCode": "string",
    "Actor2CountryCode": "string",
    "SOURCEURL": "string",
}
TITLE_CACHE_DTYPES = {"SOURCEURL": "string", "title": "string"}


def is_file_empty(path: os.PathLike[str] | str) -> bool:
    try:
        return os.path.getsize(path) == 0
    except OSError:
        return True


def is_missing_title(value: object) -> bool:
    if value is None or pd.isna(value):
        return True
    return str(value) in TITLE_MISSING_VALUES


def extract_title_from_html(content: str, fallback_url: str = "") -> str:
    soup = BeautifulSoup(content, "html.parser")
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
        if title:
            return title

    for key, value in (("property", "og:title"), ("name", "twitter:title")):
        tag = soup.find("meta", attrs={key: value})
        if tag and tag.get("content"):
            title = tag["content"].strip()
            if title:
                return title

    heading = soup.find("h1")
    if heading:
        title = heading.get_text(" ", strip=True)
        if title:
            return title

    parsed = urlparse(fallback_url)
    if parsed.netloc:
        return f"[no-title] {parsed.netloc}"
    return "[no-title]"


def write_html_file(path: os.PathLike[str] | str, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, part_path = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".part", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(part_path, target)
    finally:
        try:
            os.remove(part_path)
        except FileNotFoundError:
            pass


def write_html(base_path: os.PathLike[str] | str, content: str) -> None:
    """Compatibility wrapper for callers that pass a path without `.html`."""
    write_html_file(f"{os.fspath(base_path)}.html", content)


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


async def write_html_file_async(path: os.PathLike[str] | str, content: str) -> None:
    await asyncio.to_thread(write_html_file, path, content)


def ensure_unzipped(
    zip_path: os.PathLike[str] | str,
    file_path: os.PathLike[str] | str,
    df_path: os.PathLike[str] | str,
) -> bool:
    if is_file_empty(zip_path):
        return False

    output_dir = Path(file_path)
    target = Path(df_path)
    expected_name = target.name
    with zipfile.ZipFile(zip_path) as archive:
        member = next(
            (
                info
                for info in archive.infolist()
                if Path(info.filename).name == expected_name
            ),
            None,
        )
        if member is None:
            raise KeyError(f"Expected CSV not found in zip: {expected_name}")
        if target.exists() and member.file_size > 0 and target.stat().st_size == member.file_size:
            return True

        output_dir.mkdir(parents=True, exist_ok=True)
        descriptor, part_path = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".part", dir=output_dir
        )
        os.close(descriptor)
        try:
            with archive.open(member) as source, open(part_path, "wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            extracted_size = os.path.getsize(part_path)
            if extracted_size <= 0 or extracted_size != member.file_size:
                raise zipfile.BadZipFile(
                    "Extracted CSV size mismatch: "
                    f"expected {member.file_size}, got {extracted_size}"
                )
            os.replace(part_path, target)
        finally:
            try:
                os.remove(part_path)
            except FileNotFoundError:
                pass
    return True


def load_urls_df(
    df_path: os.PathLike[str] | str,
    columns_name_events: list[str],
    filter_country_coded: bool = True,
) -> pd.DataFrame:
    frame = pd.read_csv(
        df_path,
        sep="\t",
        names=columns_name_events,
        header=None,
        usecols=REQUIRED_EVENT_COLUMNS,
        dtype=EVENT_DTYPES,
        low_memory=False,
    )
    frame = frame[frame["SOURCEURL"].notna()]
    if filter_country_coded:
        frame = frame[
            ~(frame["Actor1CountryCode"].isna() & frame["Actor2CountryCode"].isna())
        ]

    urls = (
        frame.groupby("SOURCEURL", sort=True)["GLOBALEVENTID"]
        .apply(lambda values: [None if pd.isna(value) else int(value) for value in values])
        .reset_index()
        .reset_index()
    )
    urls["title"] = ""
    hash_paths = urls["SOURCEURL"].map(html_relative_path_for_url)
    urls["html_key"] = hash_paths.map(lambda item: item[0])
    urls["html_path"] = hash_paths.map(lambda item: item[1])
    return urls


def restore_titles_from_csv(
    urls_df: pd.DataFrame,
    csv_path: os.PathLike[str] | str,
) -> pd.DataFrame:
    if is_file_empty(csv_path):
        return urls_df
    try:
        existing = pd.read_csv(
            csv_path,
            usecols=["SOURCEURL", "title"],
            dtype=TITLE_CACHE_DTYPES,
        )
    except Exception as exc:
        logging.warning(
            "Failed to load existing title cache %s: %s",
            csv_path,
            exc.__class__.__name__,
        )
        return urls_df
    if existing.empty:
        return urls_df

    existing = existing.dropna(subset=["SOURCEURL"]).drop_duplicates(
        subset=["SOURCEURL"], keep="last"
    )
    title_map = existing.set_index("SOURCEURL")["title"]
    for index in urls_df.index:
        if not is_missing_title(urls_df.at[index, "title"]):
            continue
        restored = title_map.get(urls_df.at[index, "SOURCEURL"], "")
        if not is_missing_title(restored):
            urls_df.at[index, "title"] = restored
    return urls_df


def load_existing_hash_titles(
    urls_df: pd.DataFrame,
    zip_output_dir: os.PathLike[str] | str,
) -> pd.DataFrame:
    for index in urls_df.index:
        if not is_missing_title(urls_df.at[index, "title"]):
            continue
        key, relative_path = html_relative_path_for_url(urls_df.at[index, "SOURCEURL"])
        urls_df.at[index, "html_key"] = key
        urls_df.at[index, "html_path"] = relative_path
        html_path = absolute_html_path(zip_output_dir, relative_path)
        if is_file_empty(html_path):
            continue
        try:
            content = Path(html_path).read_text(encoding="utf-8")
            urls_df.at[index, "title"] = extract_title_from_html(
                content, str(urls_df.at[index, "SOURCEURL"])
            )
        except Exception as exc:
            logging.warning(
                "Failed to parse existing html %s: %s",
                html_path,
                exc.__class__.__name__,
            )
    return urls_df
