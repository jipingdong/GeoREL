"""Pure path and naming helpers for the full-text pipeline."""

from __future__ import annotations

import hashlib
import os


def normalize_url(value: object) -> str:
    return str(value or "").strip()


def url_hash(url: object) -> str:
    return hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()


def html_relative_path_for_url(url: object) -> tuple[str, str]:
    key = url_hash(url)
    return key, f"htmls_by_url/{key}.html"


def absolute_html_path(zip_output_dir: os.PathLike[str] | str, relative_path: str) -> str:
    return os.path.join(os.fspath(zip_output_dir), str(relative_path).replace("/", os.sep))


def extract_zip_datetime_str(zip_file: os.PathLike[str] | str) -> str:
    return os.path.basename(os.fspath(zip_file)).split(".")[0].strip()


def timestamp_year_month(timestamp: str) -> tuple[str, str]:
    return timestamp[:4], timestamp[4:6]


def zip_root(data_dir: os.PathLike[str] | str) -> str:
    return os.path.join(os.fspath(data_dir), "gdelt_zip")


def unzip_root(data_dir: os.PathLike[str] | str) -> str:
    return os.path.join(os.fspath(data_dir), "crawl_output")


def is_timestamp_zip(filename: os.PathLike[str] | str) -> bool:
    name = os.path.basename(os.fspath(filename))
    timestamp = extract_zip_datetime_str(name)
    return name.lower().endswith(".zip") and len(timestamp) == 14 and timestamp.isdigit()


def zip_output_paths(
    data_dir: os.PathLike[str] | str,
    zip_file: os.PathLike[str] | str,
) -> tuple[str, str]:
    datetime_str = extract_zip_datetime_str(zip_file)
    year, month = timestamp_year_month(datetime_str)
    file_path = os.path.join(unzip_root(data_dir), year, month, datetime_str)
    return file_path, os.path.join(file_path, "urls_title.csv")
