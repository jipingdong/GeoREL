# GDELT Article Retrieval

Extract news links from GDELT event-export ZIP files and fetch webpage HTML and titles. This directory has two entry points: `download_zipfiles.py` downloads the raw GDELT ZIP files, and `crawl.py` extracts links from the local ZIP files and fetches the webpages.

## Download GDELT Data

```bash
python download_zipfiles.py --data-dir <data-dir> --start <start-timestamp> --end <end-timestamp>
```

Downloaded `export` ZIP files are stored under `<data-dir>/gdelt_zip/YYYY/MM/` for the crawler to read next. Resume, and retry of failed or empty files, are supported.

## Fetch Webpages

```bash
python crawl.py --data-dir <data-dir> --year <year> --since-datetime <start-timestamp> --end-datetime <end-timestamp>
```

### Common Options

- `--data-dir`: input/output base directory; defaults to the `data/` directory under the repository root.
- `--year`, `--since-datetime`, `--end-datetime`: year and start/end timestamps.
- `--zip-selection-mode`: ZIP selection mode; see "Resume and Retry" below.
- `--wayback-mode`, `--selenium-mode`: whether to enable the Wayback / browser fallback; enabled by default.
- `--proxy`: proxy address; `--proxy=` means a direct connection.

Dates accept `YYYYMMDDHHMMSS`, `YYYYMMDD`, `YYYY-MM-DD`, or `YYYY-MM-DD HH:MM:SS`; both bounds are inclusive. Other options (concurrency, download strategy, retry file, state file, Wayback cache, browser driver, etc.) are documented in `--help` and `config.py`.

### Resume and Retry

Keep the input and `--data-dir` unchanged and rerun the same command to resume: ZIP files whose title tables are still valid and complete are skipped. Selection modes:

- `since_datetime` (default): process unsaved ZIP files, or saved ZIP files whose title tables are missing or invalid.
- `retry_csv`: select ZIP files from a user-provided CSV (with a `zip_name` column).
- `incremental`: resume by completion watermark.
