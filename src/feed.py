"""단순 RSS 수집 서비스."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import xml.etree.ElementTree as et
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.request import Request, urlopen

import yaml

from model import CrawlItem, Source
from scrap import NewsScraper


@dataclass(slots=True)
class _RssEntry:
    title: str
    link: str
    pub_date: str


class RssFetcher:
    """RSS 단건 수집기."""

    def __init__(self, count: int, scraper: NewsScraper) -> None:
        self.count = count
        self.scraper = scraper

    def get_sources(self, config_path: str = "resources/config.yml") -> list[Source]:
        """config.yml의 rss 목록을 Source 리스트로 변환한다."""
        config = _load_config(config_path)
        rss_config = config.get("rss", {})

        sources: list[Source] = []
        for reference, categories in rss_config.items():
            for category, url in categories.items():
                sources.append(Source(url=url, reference=reference or "None", category=category))
        return sources

    def fetch(self, ctx: object, source: Source) -> list[CrawlItem]:
        """소스 하나를 RSS로 수집하고 본문 스크랩까지 수행한다."""
        xml_body = _http_get(source.url)
        rss_items = _parse_rss(xml_body)

        limit = self.count if self.count > 0 else len(rss_items)
        limit = min(limit, len(rss_items))

        items: list[CrawlItem] = []
        for rss in rss_items[:limit]:
            if "/video" in rss.link:
                logging.info("rss 영상 링크 제외: url=%s", rss.link)
                continue

            try:
                published_at = _parse_time(rss.pub_date)
            except Exception:  # noqa: BLE001
                logging.info("rss 날짜 파싱 실패: url=%s", rss.link)
                continue

            content_list = self.scraper.scrap(ctx, rss.link)
            if not content_list:
                logging.info("rss 스크랩 결과 없음: url=%s", rss.link)
                continue

            items.append(
                CrawlItem(
                    title=rss.title,
                    content=content_list[0],
                    source=source,
                    url=rss.link,
                    published_at=published_at,
                )
            )

        return items


def get_feeds(
    ctx: object,
    fetcher: RssFetcher,
    config_path: str = "resources/config.yml",
    max_workers: int = 8,
) -> list[CrawlItem]:
    """전체 RSS 소스를 병렬 수집한다."""
    sources = fetcher.get_sources(config_path=config_path)
    if not sources:
        return []

    workers = min(max_workers, len(sources)) if max_workers > 0 else len(sources)

    results: list[CrawlItem] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(fetcher.fetch, ctx, source): source for source in sources}
        for future in as_completed(future_map):
            source = future_map[future]
            try:
                results.extend(future.result())
            except Exception as err:  # noqa: BLE001
                logging.info("rss 수집 실패: url=%s err=%s", source.url, err)
    return results


def _load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise ValueError(f"config not found: {config_path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    rss = config.get("rss")
    if rss is None or not isinstance(rss, dict):
        raise ValueError("invalid config: 'rss' must be a mapping")

    for reference, categories in rss.items():
        if not isinstance(categories, dict):
            raise ValueError(f"invalid config: rss.{reference} must be a mapping")
        for category, url in categories.items():
            if not isinstance(url, str) or not url:
                raise ValueError(f"invalid config: rss.{reference}.{category} must be a non-empty string")

    return config


def _http_get(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "IssueEngine-RSS/1.0"})
    with urlopen(req, timeout=30) as resp:
        return resp.read()


def _parse_rss(body: bytes) -> list[_RssEntry]:
    root = et.fromstring(body)
    channel = root.find("channel")
    if channel is None:
        return []

    items: list[_RssEntry] = []
    for item in channel.findall("item"):
        items.append(
            _RssEntry(
                title=(item.findtext("title") or "").strip(),
                link=(item.findtext("link") or "").strip(),
                pub_date=(item.findtext("pubDate") or "").strip(),
            )
        )
    return items


def _parse_time(pub_date: str) -> datetime:
    parsed = parsedate_to_datetime(pub_date)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
