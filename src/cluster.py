"""RSS 기사 클러스터링 서비스."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

from .model import IssueEmbedding, MatchCandidate
from .repository import ArticleRepository, IssueEmbeddingRepository, IssueRepository

LAMBDA_CONST = 0.1
ALPHA_CONST = 0.8
BETA_CONST = 0.2
SEPARABILITY_THRESHOLD = 0.1
HIGH_SIMILARITY_THRESHOLD = 0.73
FLOAT_EPSILON = 1.1920929e-07


class ClusterService:
    """기사를 기존 이슈에 병합하거나 새 이슈를 생성한다."""

    def __init__(
        self,
        embedding_manager: Any,
        issue_repo: IssueRepository,
        article_repo: ArticleRepository,
        issue_embedding_repo: IssueEmbeddingRepository,
    ) -> None:
        self.embedding_manager = embedding_manager
        self.issue_repo = issue_repo
        self.article_repo = article_repo
        self.issue_embedding_repo = issue_embedding_repo

    def run(self, ctx: object) -> int:
        """이슈 미매핑 기사 전체를 처리한다."""
        articles = self.article_repo.list_by_issue_id(ctx, 0)
        processed = 0
        for item in articles:
            self.cluster(ctx, item.title, item.content, item.id)
            processed += 1
        return processed

    def cluster(self, ctx: object, title: str, content: str, article_id: int) -> int | None:
        """단일 기사를 클러스터링한다."""
        embedding_res = self.embedding_manager.generate(ctx, title, content)

        article_vector = list(embedding_res.embedding_feature.dense)
        if not article_vector:
            raise ValueError("embedding vector is empty")
        _normalize(article_vector)

        candidates = self.find_best_cluster(ctx, article_vector, datetime.now(timezone.utc))
        best = candidates[0] if candidates else None
        neighbor_similarity = candidates[1].similarity if len(candidates) > 1 else None

        target_issue_id = self._resolve_target_issue(title, best, neighbor_similarity)
        if target_issue_id is not None:
            self.update_cluster(
                ctx,
                target_issue_id,
                article_vector,
            )
            self.article_repo.update_issue_id(ctx, article_id, target_issue_id)
            return None

        issue_id = self.issue_repo.create(ctx, title, "not yet summary", 1)
        self.issue_embedding_repo.create(
            ctx,
            IssueEmbedding(
                issue_id=issue_id,
                dense=article_vector,
            ),
        )
        self.article_repo.update_issue_id(ctx, article_id, issue_id)
        return issue_id

    def find_best_cluster(
        self,
        ctx: object,
        article_vec: list[float],
        now: datetime,
    ) -> list[MatchCandidate]:
        """유사한 이슈 후보를 찾아 유사도 기준으로 정렬한다."""
        issue_ids = self.issue_embedding_repo.find_similar_issue_ids(ctx, article_vec)

        candidates: list[MatchCandidate] = []
        for issue_id in issue_ids:
            embedding_item = self.issue_embedding_repo.find_by_issue_id(ctx, issue_id)
            similarity = _dot_product(article_vec, embedding_item.dense)

            issue_item = self.issue_repo.find_by_id(ctx, issue_id)
            hours_diff = (now - issue_item.updated_at).total_seconds() / 3600
            if hours_diff < 0:
                hours_diff = 0

            weighted_time = math.exp(-LAMBDA_CONST * hours_diff)
            score = (ALPHA_CONST * similarity) + (BETA_CONST * weighted_time)

            candidates.append(
                MatchCandidate(
                    issue_id=issue_id,
                    title=issue_item.title,
                    score=score,
                    similarity=similarity,
                )
            )

        candidates.sort(key=lambda item: item.similarity, reverse=True)
        return candidates

    def update_cluster(
        self,
        ctx: object,
        issue_id: int,
        new_vec: list[float],
    ) -> None:
        """기존 클러스터 중심점을 갱신한다."""
        embedding_item = self.issue_embedding_repo.find_by_issue_id(ctx, issue_id)
        issue_item = self.issue_repo.find_by_id(ctx, issue_id)

        count = float(issue_item.article_count)
        centroid = list(embedding_item.dense)
        for idx, value in enumerate(centroid):
            centroid[idx] = ((value * count) + new_vec[idx]) / (count + 1.0)
        _normalize(centroid)

        self.issue_embedding_repo.update(ctx, issue_id, centroid)
        self.issue_repo.update(ctx, issue_id)

    def _resolve_target_issue(
        self,
        title: str,
        best: MatchCandidate | None,
        neighbor_similarity: float | None,
    ) -> int | None:
        """분리도 규칙으로 병합 대상 이슈를 결정한다."""
        if best is None:
            logging.info("초기 상태")
            return None

        if best.similarity < HIGH_SIMILARITY_THRESHOLD:
            logging.info("새 클러스터 생성", extra={"reason": "low_similarity", "input_title": title})
            return None

        neighbor = neighbor_similarity if neighbor_similarity is not None else 0.0
        if neighbor < FLOAT_EPSILON:
            logging.info("기존 클러스터 병합", extra={"reason": "dominant_best", "input_title": title})
            return best.issue_id

        a = 1.0 - best.similarity
        b = 1.0 - neighbor
        denominator = max(a, b)
        separability = 0.0
        if denominator >= FLOAT_EPSILON:
            separability = (b - a) / denominator

        if separability > SEPARABILITY_THRESHOLD:
            logging.info("기존 클러스터 병합", extra={"reason": "high_separability", "input_title": title})
            return best.issue_id

        logging.info("새 클러스터 생성", extra={"reason": "low_separability", "input_title": title})
        return None


def _normalize(values: list[float]) -> None:
    magnitude = math.sqrt(sum(v * v for v in values))
    if magnitude <= FLOAT_EPSILON:
        return
    for idx, value in enumerate(values):
        values[idx] = value / magnitude


def _dot_product(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
