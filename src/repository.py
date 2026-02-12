"""Cluster 서비스용 Postgres 저장소 구현."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol


@dataclass(slots=True)
class ArticleRow:
    id: int
    title: str
    content: str


@dataclass(slots=True)
class IssueRow:
    id: int
    title: str
    updated_at: datetime
    article_count: int


@dataclass(slots=True)
class IssueEmbeddingRow:
    issue_id: int
    dense: list[float]


class IssueRepository(Protocol):
    def create(self, ctx: object, title: str, summary: str, status: int) -> int:
        """새 이슈를 생성하고 식별자를 반환한다."""

    def find_by_id(self, ctx: object, issue_id: int) -> IssueRow:
        """이슈 단건을 조회한다."""

    def update(self, ctx: object, issue_id: int) -> None:
        """이슈의 article_count 와 updated_at 을 갱신한다."""


class ArticleRepository(Protocol):
    def list_by_issue_id(self, ctx: object, issue_id: int) -> list[ArticleRow]:
        """특정 issue_id 를 가진 기사 목록을 조회한다."""

    def update_issue_id(self, ctx: object, article_id: int, issue_id: int) -> None:
        """기사의 issue_id 를 변경한다."""


class IssueEmbeddingRepository(Protocol):
    def create(self, ctx: object, item: Any) -> None:
        """이슈 임베딩을 생성한다."""

    def find_similar_issue_ids(self, ctx: object, dense: list[float]) -> list[int]:
        """입력 벡터와 유사한 이슈 ID 목록을 조회한다."""

    def find_by_issue_id(self, ctx: object, issue_id: int) -> IssueEmbeddingRow:
        """이슈 임베딩 단건을 조회한다."""

    def update(self, ctx: object, issue_id: int, dense: list[float]) -> None:
        """이슈 임베딩을 갱신한다."""


class PostgresIssueRepository:
    def __init__(self, conn: Any, table: str = "issues") -> None:
        self._conn = conn
        self._table = table

    def create(self, ctx: object, title: str, summary: str, status: int) -> int:
        now = datetime.now(timezone.utc)
        sql = f"""
            INSERT INTO {self._table} (title, summary, status, article_count, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        with _cursor(self._resolve_conn(ctx)) as cur:
            cur.execute(sql, (title, summary, status, 1, now, now))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("failed to create issue")
        return int(row[0])

    def find_by_id(self, ctx: object, issue_id: int) -> IssueRow:
        sql = f"""
            SELECT id, title, updated_at, article_count
            FROM {self._table}
            WHERE id = %s
        """
        with _cursor(self._resolve_conn(ctx)) as cur:
            cur.execute(sql, (issue_id,))
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"issue not found: {issue_id}")
        return IssueRow(
            id=int(row[0]),
            title=str(row[1]),
            updated_at=row[2],
            article_count=int(row[3]),
        )

    def update(self, ctx: object, issue_id: int) -> None:
        sql = f"""
            UPDATE {self._table}
            SET article_count = article_count + 1,
                updated_at = %s
            WHERE id = %s
        """
        with _cursor(self._resolve_conn(ctx)) as cur:
            cur.execute(sql, (datetime.now(timezone.utc), issue_id))
            if cur.rowcount == 0:
                raise KeyError(f"issue not found: {issue_id}")

    def _resolve_conn(self, ctx: object) -> Any:
        return ctx if ctx is not None else self._conn


class PostgresArticleRepository:
    def __init__(self, conn: Any, table: str = "articles") -> None:
        self._conn = conn
        self._table = table

    def list_by_issue_id(self, ctx: object, issue_id: int) -> list[ArticleRow]:
        sql = f"""
            SELECT id, title, content
            FROM {self._table}
            WHERE issue_id = %s
            ORDER BY id ASC
        """
        with _cursor(self._resolve_conn(ctx)) as cur:
            cur.execute(sql, (issue_id,))
            rows = cur.fetchall()
        return [ArticleRow(id=int(row[0]), title=str(row[1]), content=str(row[2])) for row in rows]

    def update_issue_id(self, ctx: object, article_id: int, issue_id: int) -> None:
        sql = f"""
            UPDATE {self._table}
            SET issue_id = %s
            WHERE id = %s
        """
        with _cursor(self._resolve_conn(ctx)) as cur:
            cur.execute(sql, (issue_id, article_id))
            if cur.rowcount == 0:
                raise KeyError(f"article not found: {article_id}")

    def _resolve_conn(self, ctx: object) -> Any:
        return ctx if ctx is not None else self._conn


class PostgresIssueEmbeddingRepository:
    def __init__(
        self,
        conn: Any,
        table: str = "issue_embeddings",
        similarity_limit: int = 20,
    ) -> None:
        self._conn = conn
        self._table = table
        self._similarity_limit = similarity_limit

    def create(self, ctx: object, item: Any) -> None:
        sql = f"""
            INSERT INTO {self._table} (issue_id, dense)
            VALUES (%s, %s::vector)
        """
        with _cursor(self._resolve_conn(ctx)) as cur:
            cur.execute(sql, (int(item.issue_id), _to_pgvector_literal(item.dense)))

    def find_similar_issue_ids(self, ctx: object, dense: list[float]) -> list[int]:
        sql = f"""
            SELECT issue_id
            FROM {self._table}
            ORDER BY dense <=> %s::vector ASC
            LIMIT %s
        """
        with _cursor(self._resolve_conn(ctx)) as cur:
            cur.execute(sql, (_to_pgvector_literal(dense), self._similarity_limit))
            rows = cur.fetchall()
        return [int(row[0]) for row in rows]

    def find_by_issue_id(self, ctx: object, issue_id: int) -> IssueEmbeddingRow:
        sql = f"""
            SELECT issue_id, dense
            FROM {self._table}
            WHERE issue_id = %s
        """
        with _cursor(self._resolve_conn(ctx)) as cur:
            cur.execute(sql, (issue_id,))
            row = cur.fetchone()
            if row is None:
                raise KeyError(f"issue embedding not found: {issue_id}")
        return IssueEmbeddingRow(issue_id=int(row[0]), dense=_from_vector_value(row[1]))

    def update(self, ctx: object, issue_id: int, dense: list[float]) -> None:
        sql = f"""
            UPDATE {self._table}
            SET dense = %s::vector
            WHERE issue_id = %s
        """
        with _cursor(self._resolve_conn(ctx)) as cur:
            cur.execute(sql, (_to_pgvector_literal(dense), issue_id))
            if cur.rowcount == 0:
                raise KeyError(f"issue embedding not found: {issue_id}")

    def _resolve_conn(self, ctx: object) -> Any:
        return ctx if ctx is not None else self._conn


def _to_pgvector_literal(dense: list[float]) -> str:
    if not dense:
        raise ValueError("dense vector is empty")
    return "[" + ",".join(f"{float(v):.12g}" for v in dense) + "]"


def _from_vector_value(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, tuple):
        return [float(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            body = text[1:-1].strip()
            if not body:
                return []
            return [float(part.strip()) for part in body.split(",")]
    raise TypeError(f"unsupported vector value type: {type(value)!r}")


@contextmanager
def _cursor(conn: Any):
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
