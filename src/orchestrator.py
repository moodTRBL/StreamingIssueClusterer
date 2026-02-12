from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path

import psycopg2
import yaml

from cluster import ClusterService
from embedding import Vectorizer
from feed import RssFetcher, get_feeds
from repository import (
    PostgresArticleRepository,
    PostgresIssueEmbeddingRepository,
    PostgresIssueRepository,
)
from scrap import NewsScraper


@dataclass(slots=True)
class PipelineResult:
    scraped: int
    saved: int
    clustered: int


class PipelineOrchestrator:
    def __init__(self, config_path: str = "resources/config.yml") -> None:
        self._config_path = Path(config_path)

    def run(self) -> PipelineResult:
        count, workers, dsn = self._load_config()
        if not dsn:
            raise ValueError("database dsn is required. set db.dsn in resources/config.yml or DATABASE_URL")

        fetcher = RssFetcher(count=count, scraper=NewsScraper())
        items = get_feeds(
            ctx=None,
            fetcher=fetcher,
            config_path=str(self._config_path),
            max_workers=workers,
        )

        conn = psycopg2.connect(dsn)
        try:
            article_repo = PostgresArticleRepository(conn)
            issue_repo = PostgresIssueRepository(conn)
            issue_embedding_repo = PostgresIssueEmbeddingRepository(conn)

            saved = 0
            for item in items:
                source_name = f"{item.source.reference}/{item.source.category}"
                article_repo.create(
                    ctx=conn,
                    title=item.title,
                    content=item.content,
                    source=source_name,
                    url=item.url,
                    published_at=item.published_at,
                )
                saved += 1

            cluster_service = ClusterService(
                embedding_manager=Vectorizer(),
                issue_repo=issue_repo,
                article_repo=article_repo,
                issue_embedding_repo=issue_embedding_repo,
            )
            clustered = cluster_service.run(conn)
        finally:
            conn.close()

        logging.info("pipeline done: scraped=%s saved=%s clustered=%s", len(items), saved, clustered)
        return PipelineResult(scraped=len(items), saved=saved, clustered=clustered)

    def _load_config(self) -> tuple[int, int, str]:
        if not self._config_path.exists():
            raise ValueError(f"config not found: {self._config_path}")

        with self._config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

        run = config.get("run", {})
        count = int(run.get("count", 3))
        workers = int(run.get("workers", 8))

        db = config.get("db", {})
        config_dsn = str(db.get("dsn", "")).strip()
        dsn = os.getenv("DATABASE_URL", config_dsn).strip()
        return count, workers, dsn
