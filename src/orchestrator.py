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
    IssueRow,
    PostgresArticleRepository,
    PostgresIssueEmbeddingRepository,
    PostgresIssueRepository,
    ensure_schema,
)
from scrap import NewsScraper


@dataclass(slots=True)
class PipelineResult:
    scraped: int
    saved: int
    clustered: int
    issues: list[IssueRow]


class PipelineOrchestrator:
    def __init__(self, config_path: str = "resources/config.yml") -> None:
        self.config_path = Path(config_path)
        self.count, self.workers = self.load_config()

        dsn = os.getenv("DATABASE_URL", "").strip()
        if not dsn:
            raise ValueError("database dsn is required. set DATABASE_URL or .env values")

        self.scraper = NewsScraper()
        self.fetcher = RssFetcher(count=self.count, scraper=self.scraper)

        self.conn = psycopg2.connect(dsn)
        ensure_schema(self.conn)
        self.article_repo = PostgresArticleRepository(self.conn)
        self.issue_repo = PostgresIssueRepository(self.conn)
        self.issue_embedding_repo = PostgresIssueEmbeddingRepository(self.conn)
        self.vectorizer = Vectorizer()
        self.cluster_service = ClusterService(
            embedding_manager=self.vectorizer,
            issue_repo=self.issue_repo,
            article_repo=self.article_repo,
            issue_embedding_repo=self.issue_embedding_repo,
        )

    def run(self) -> PipelineResult:
        items = get_feeds(
            ctx=None,
            fetcher=self.fetcher,
            config_path=str(self.config_path),
            max_workers=self.workers,
        )

        try:
            saved = 0
            for item in items:
                source_name = f"{item.source.reference}/{item.source.category}"
                self.article_repo.create(
                    ctx=self.conn,
                    title=item.title,
                    content=item.content,
                    source=source_name,
                    url=item.url,
                    published_at=item.published_at,
                )
                saved += 1

            clustered = self.cluster_service.run(self.conn)
            issues = self.issue_repo.list_all(self.conn)
        finally:
            self.conn.close()

        logging.info("pipeline done: scraped=%s saved=%s clustered=%s", len(items), saved, clustered)
        return PipelineResult(scraped=len(items), saved=saved, clustered=clustered, issues=issues)

    def load_config(self) -> tuple[int, int]:
        if not self.config_path.exists():
            raise ValueError(f"config not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

        run = config.get("run", {})
        count = int(run.get("count", 3))
        workers = int(run.get("workers", 8))
        return count, workers
