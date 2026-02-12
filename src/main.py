from __future__ import annotations

import logging
import os
from pathlib import Path

import psycopg2
import yaml

from feed import RssFetcher, get_feeds
from repository import PostgresArticleRepository
from scrap import NewsScraper

CONFIG_PATH = Path("resources/config.yml")


def _load_run_config(path: Path) -> tuple[int, int, str]:
    if not path.exists():
        raise ValueError(f"config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    run = config.get("run", {})
    count = int(run.get("count", 3))
    workers = int(run.get("workers", 8))
    db = config.get("db", {})
    dsn = str(db.get("dsn", "")).strip()
    return count, workers, dsn


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    count, workers, config_dsn = _load_run_config(CONFIG_PATH)
    dsn = os.getenv("DATABASE_URL", config_dsn).strip()
    if not dsn:
        raise ValueError("database dsn is required. set db.dsn in resources/config.yml or DATABASE_URL")

    fetcher = RssFetcher(count=count, scraper=NewsScraper())
    items = get_feeds(
        ctx=None,
        fetcher=fetcher,
        config_path=str(CONFIG_PATH),
        max_workers=workers,
    )

    conn = psycopg2.connect(dsn)
    article_repo = PostgresArticleRepository(conn)
    saved = 0
    for item in items:
        source_name = f"{item.source.reference}/{item.source.category}"
        article_repo.create(
            ctx=None,
            title=item.title,
            content=item.content,
            source=source_name,
            url=item.url,
            published_at=item.published_at,
        )
        saved += 1
    conn.close()

    print(f"scraped items: {len(items)}")
    print(f"saved items: {saved}")
    for idx, item in enumerate(items, start=1):
        preview = item.content[:120].replace("\n", " ")
        print(
            f"[{idx}] [{item.source.url}/{item.source.category}] "
            f"{item.title} ({item.published_at.isoformat()})"
        )
        print(f"    {preview}...")


if __name__ == "__main__":
    main()
