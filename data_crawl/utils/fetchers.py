"""Asynchronous stage scheduling and page fetch implementations."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from contextlib import AsyncExitStack, asynccontextmanager
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

from data_crawl.config import RuntimeConfig
from data_crawl.utils.locks import acquire_slot, release_slot
from data_crawl.utils.paths import absolute_html_path
from data_crawl.utils.storage import extract_title_from_html, is_missing_title, write_html_file_async
from data_crawl.utils.wayback import WaybackCache, get_wayback_snapshot


StageFunction = Callable[[Any, str, RuntimeConfig], Awaitable[tuple[int, str]]]
WAYBACK_AVAILABILITY_ENDPOINTS = (
    "https://archive.org/wayback/available",
    "https://web.archive.org/wayback/available",
)
WAYBACK_CDX_API = "https://web.archive.org/cdx/search/cdx"
REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=20.0, write=20.0, pool=20.0)
WAYBACK_TIMEOUT = httpx.Timeout(connect=10.0, read=15.0, write=15.0, pool=15.0)
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
}


class Crawler:
    def __init__(self, config: RuntimeConfig | None = None) -> None:
        if config is None:
            driver_path = str(Path(__file__).resolve().parent.parent / "chromedriver.exe")
            chrome_binary = ""
            proxy = "http://127.0.0.1:7900"
        else:
            driver_path = config.chromedriver_path
            chrome_binary = config.chrome_binary_path
            proxy = config.proxy
        if not driver_path or not os.path.isfile(driver_path):
            raise FileNotFoundError(f"ChromeDriver not found: {driver_path}")

        options = webdriver.ChromeOptions()
        if chrome_binary:
            options.binary_location = chrome_binary
        for argument in (
            "--disable-blink-features=AutomationControlled",
            "--incognito",
            f"user-agent={DEFAULT_HEADERS['User-Agent']}",
            "--no-sandbox",
            "--window-size=1920,1080",
            "--disable-gpu",
            "--disable-3d-apis",
            "--disable-dev-shm-usage",
            "--hide-scrollbars",
            "--headless=new",
            "--remote-debugging-pipe",
            "test-type",
            "log-level=3",
        ):
            options.add_argument(argument)
        proxy_route = direct_routes(proxy)[0]
        if proxy_route:
            options.add_argument(f"--proxy-server={proxy_route}")
        options.add_experimental_option(
            "excludeSwitches",
            ["enable-logging", "ignore-certificate-errors", "enable-automation"],
        )
        self.driver = webdriver.Chrome(
            service=Service(
                executable_path=driver_path,
                log_output=subprocess.DEVNULL,
            ),
            options=options,
        )

    def fetch_html(self, url: str) -> str:
        self.driver.get(url)
        return self.driver.page_source

    def cleanup(self) -> None:
        driver = getattr(self, "driver", None)
        if driver is not None:
            driver.quit()


def direct_routes(proxy: object) -> tuple[str | None, ...]:
    value = str(proxy or "").strip()
    if not value:
        return (None,)
    if "://" not in value:
        value = f"http://{value}"
    return value, None


def _ordered_stage_names(config: RuntimeConfig) -> list[str]:
    primary = (
        ["wayback", "direct"]
        if config.download_strategy == "wayback_first"
        else ["direct", "wayback"]
    )
    enabled = [
        name
        for name in primary
        if name != "wayback" or config.wayback_mode == "on"
    ]
    if config.selenium_mode == "on":
        enabled.append("selenium")
    return enabled


def _stage_concurrency(stage_name: str, config: RuntimeConfig) -> int:
    if stage_name == "wayback":
        return config.wayback_concurrency
    if stage_name == "selenium":
        return config.selenium_max_concurrency
    return config.url_concurrency


async def _apply_stage(
    urls_df: pd.DataFrame,
    file_path: str,
    config: RuntimeConfig,
    stage_name: str,
    stage_function: StageFunction,
) -> None:
    pending = urls_df[urls_df["title"].map(is_missing_title)]
    if pending.empty:
        return
    semaphore = asyncio.Semaphore(max(1, _stage_concurrency(stage_name, config)))

    async def guarded(row):
        async with semaphore:
            return await stage_function(row, file_path, config)

    rows = list(pending.itertuples(index=False))
    batch_size = max(1, config.async_batch_size)
    for offset in range(0, len(rows), batch_size):
        results = await asyncio.gather(
            *(guarded(row) for row in rows[offset : offset + batch_size]),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                logging.warning(
                    "%s stage row failed: %s", stage_name, result.__class__.__name__
                )
                continue
            row_index, title = result
            if not is_missing_title(title):
                urls_df.loc[row_index, "title"] = title


async def fetch_all_async(
    urls_df: pd.DataFrame,
    file_path: str,
    config: RuntimeConfig,
    stage_functions: Mapping[str, StageFunction] | None = None,
) -> pd.DataFrame:
    if stage_functions is None:
        async with default_stage_functions(config) as default_functions:
            return await fetch_all_async(urls_df, file_path, config, default_functions)
    result = urls_df.copy()
    for stage_name in _ordered_stage_names(config):
        await _apply_stage(
            result,
            file_path,
            config,
            stage_name,
            stage_functions[stage_name],
        )
    return result


async def fetch_wayback_row(
    row: Any,
    file_path: str,
    config: RuntimeConfig,
    *,
    snapshot_resolver: Callable[[str], Awaitable[str]] | None = None,
    archive_fetcher: Callable[[str], Awaitable[str]] | None = None,
    acquire_slot_fn: Callable[..., str | None] = acquire_slot,
    release_slot_fn: Callable[[str | None], None] = release_slot,
) -> tuple[int, str]:
    slot = await asyncio.to_thread(
        acquire_slot_fn,
        config.wayback_slot_dir,
        "wayback",
        config.wayback_global_concurrency,
        config.wayback_lock_timeout,
        config.wayback_lock_stale_seconds,
    )
    if slot is None:
        return row.index, ""
    try:
        if snapshot_resolver is None or archive_fetcher is None:
            raise RuntimeError("Wayback fetch dependencies are not configured")
        archive_url = await snapshot_resolver(str(row.SOURCEURL))
        if not archive_url:
            return row.index, ""
        content = await archive_fetcher(archive_url)
        title = extract_title_from_html(content, archive_url)
        await write_html_file_async(
            absolute_html_path(file_path, row.html_path), content
        )
        return row.index, title
    except Exception as exc:
        logging.warning(
            "Wayback stage failed for %s: %s",
            row.SOURCEURL,
            exc.__class__.__name__,
        )
        return row.index, ""
    finally:
        release_slot_fn(slot)


def build_async_client(
    route: str | None,
    concurrency: int,
) -> httpx.AsyncClient:
    limits = httpx.Limits(
        max_connections=max(int(concurrency) + 16, 32),
        max_keepalive_connections=max(min(int(concurrency), 32), 16),
    )
    kwargs: dict[str, Any] = {
        "headers": DEFAULT_HEADERS,
        "follow_redirects": True,
        "limits": limits,
        "timeout": REQUEST_TIMEOUT,
        "trust_env": False,
    }
    if route:
        kwargs["proxy"] = route
    return httpx.AsyncClient(**kwargs)


def _is_url(value: object) -> bool:
    try:
        parsed = urlparse(str(value))
        return bool(parsed.scheme and parsed.netloc)
    except ValueError:
        return False


async def fetch_direct_row(
    row: Any,
    file_path: str,
    config: RuntimeConfig,
    *,
    clients: Mapping[str | None, Any],
) -> tuple[int, str]:
    url = str(row.SOURCEURL)
    if not _is_url(url):
        return row.index, url
    failures: list[str] = []
    for route in direct_routes(config.proxy):
        client = clients[route]
        try:
            response = await client.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            content = response.text
            title = extract_title_from_html(content, str(response.url))
            await write_html_file_async(
                absolute_html_path(file_path, row.html_path), content
            )
            return row.index, title
        except Exception as exc:
            failures.append(exc.__class__.__name__)
    if failures:
        logging.warning("Direct stage failed for %s: %s", url, ",".join(failures))
    return row.index, ""


async def _get_json(client: Any, url: str, params: dict[str, Any]) -> Any:
    response = await client.get(url, timeout=WAYBACK_TIMEOUT, params=params)
    response.raise_for_status()
    return response.json()


async def lookup_wayback_snapshot(
    source_url: str,
    clients: Mapping[str | None, Any],
) -> str:
    had_response = False
    failures: list[Exception] = []
    for endpoint in WAYBACK_AVAILABILITY_ENDPOINTS:
        for client in clients.values():
            try:
                payload = await _get_json(client, endpoint, {"url": source_url})
                had_response = True
                closest = payload.get("archived_snapshots", {}).get("closest", {})
                if closest.get("available") and closest.get("url"):
                    return str(closest["url"])
            except Exception as exc:
                failures.append(exc)

    cdx_params = {
        "url": source_url,
        "output": "json",
        "fl": "timestamp,original,statuscode",
        "filter": "statuscode:200",
        "limit": 1,
        "sort": "reverse",
    }
    for client in clients.values():
        try:
            rows = await _get_json(client, WAYBACK_CDX_API, cdx_params)
            had_response = True
            if isinstance(rows, list) and len(rows) >= 2 and len(rows[1]) >= 2:
                timestamp, original = rows[1][0], rows[1][1]
                return f"https://web.archive.org/web/{timestamp}/{original}"
        except Exception as exc:
            failures.append(exc)
    if failures and not had_response:
        raise RuntimeError("all Wayback lookup endpoints failed") from failures[-1]
    return ""


async def fetch_archive_html(
    archive_url: str,
    clients: Mapping[str | None, Any],
) -> str:
    failures: list[Exception] = []
    for client in clients.values():
        try:
            response = await client.get(archive_url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            failures.append(exc)
    raise RuntimeError("all Wayback archive downloads failed") from failures[-1]


def _run_crawler(
    url: str,
    config: RuntimeConfig,
    crawler_factory: Callable[[RuntimeConfig], Any],
) -> str:
    crawler = crawler_factory(config)
    try:
        return crawler.fetch_html(url)
    finally:
        crawler.cleanup()


async def fetch_selenium_row(
    row: Any,
    file_path: str,
    config: RuntimeConfig,
    *,
    crawler_factory: Callable[[RuntimeConfig], Any] = Crawler,
    acquire_slot_fn: Callable[..., str | None] = acquire_slot,
    release_slot_fn: Callable[[str | None], None] = release_slot,
) -> tuple[int, str]:
    slot = await asyncio.to_thread(
        acquire_slot_fn,
        config.selenium_slot_dir,
        "selenium",
        config.selenium_max_concurrency,
        config.selenium_lock_timeout,
        config.wayback_lock_stale_seconds,
    )
    if slot is None:
        return row.index, ""
    try:
        content = await asyncio.to_thread(
            _run_crawler, str(row.SOURCEURL), config, crawler_factory
        )
        title = extract_title_from_html(content, str(row.SOURCEURL))
        await write_html_file_async(
            absolute_html_path(file_path, row.html_path), content
        )
        return row.index, title
    except Exception as exc:
        logging.warning(
            "Selenium stage failed for %s: %s",
            row.SOURCEURL,
            exc.__class__.__name__,
        )
        return row.index, ""
    finally:
        release_slot_fn(slot)


@asynccontextmanager
async def default_stage_functions(config: RuntimeConfig):
    async with AsyncExitStack() as stack:
        clients: dict[str | None, httpx.AsyncClient] = {}
        for route in direct_routes(config.proxy):
            clients[route] = await stack.enter_async_context(
                build_async_client(route, config.url_concurrency)
            )
        cache = WaybackCache(
            config.wayback_cache_path,
            config.wayback_hit_ttl_seconds,
            config.wayback_miss_ttl_seconds,
            config.wayback_error_ttl_seconds,
        )

        async def direct(row, file_path, runtime_config):
            return await fetch_direct_row(
                row, file_path, runtime_config, clients=clients
            )

        async def resolve_snapshot(url: str) -> str:
            async def lookup(value: str) -> str:
                return await lookup_wayback_snapshot(value, clients)

            return await get_wayback_snapshot(url, cache, lookup)

        async def download_archive(url: str) -> str:
            return await fetch_archive_html(url, clients)

        async def wayback(row, file_path, runtime_config):
            return await fetch_wayback_row(
                row,
                file_path,
                runtime_config,
                snapshot_resolver=resolve_snapshot,
                archive_fetcher=download_archive,
            )

        async def selenium(row, file_path, runtime_config):
            return await fetch_selenium_row(row, file_path, runtime_config)

        yield {"direct": direct, "wayback": wayback, "selenium": selenium}
