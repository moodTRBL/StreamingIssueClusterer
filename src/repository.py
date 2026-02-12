"""Cluster 서비스용 Postgres 저장소 구현."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
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


def _load_dotenv_file(path: str = ".env") -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        raise ValueError(".env file not found")

    values: dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _ensure_database_url_from_dotenv(path: str = ".env") -> None:
    if os.getenv("DATABASE_URL", "").strip():
        return

    values = _load_dotenv_file(path)
    required = ["DB_HOST", "DB_PORT", "DB_USER", "DB_PASSWORD", "DB_NAME", "DB_SSLMODE"]
    missing = [key for key in required if not values.get(key, "").strip()]
    if missing:
        raise ValueError(f"missing required .env values: {', '.join(missing)}")

    dsn = (
        f"host={values['DB_HOST']} "
        f"port={values['DB_PORT']} "
        f"user={values['DB_USER']} "
        f"password={values['DB_PASSWORD']} "
        f"dbname={values['DB_NAME']} "
        f"sslmode={values['DB_SSLMODE']}"
    )
    os.environ["DATABASE_URL"] = dsn


_ensure_database_url_from_dotenv()


def ensure_schema(conn: Any) -> None:
    schema_path = Path(__file__).resolve().parent.parent / "resources" / "schema.sql"
    if not schema_path.exists():
        raise ValueError(f"schema file not found: {schema_path}")

    sql = schema_path.read_text(encoding="utf-8")
    with _cursor(conn) as cur:
        cur.execute(sql)
        cur.execute(
            """
            ALTER TABLE issue_embedding
                DROP COLUMN IF EXISTS sparse_indice,
                DROP COLUMN IF EXISTS sparse_value
            """
        )


class IssueRepository(Protocol):
    def create(self, ctx: object, title: str, summary: str, status: int) -> int:
        """새 이슈를 생성하고 식별자를 반환한다."""

    def find_by_id(self, ctx: object, issue_id: int) -> IssueRow:
        """이슈 단건을 조회한다."""

    def update(self, ctx: object, issue_id: int) -> None:
        """이슈의 article_count 와 updated_at 을 갱신한다."""

    def list_all(self, ctx: object) -> list[IssueRow]:
        """이슈 전체 목록을 조회한다."""


class ArticleRepository(Protocol):
    def create(
        self,
        ctx: object,
        title: str,
        content: str,
        source: str,
        url: str,
        published_at: datetime | None,
    ) -> int:
        """기사를 생성하고 식별자를 반환한다."""

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
    def __init__(self, conn: Any, table: str = "issue") -> None:
        self._conn = conn
        self._table = table

    def create(self, ctx: object, title: str, summary: str, status: int) -> int:
        del status
        now = datetime.now(timezone.utc)
        sql = f"""
            INSERT INTO {self._table} (title, content, article_count, started_at, updated_at, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        with _cursor(self._resolve_conn(ctx)) as cur:
            cur.execute(sql, (title, summary, 1, now, now, now))
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

    def list_all(self, ctx: object) -> list[IssueRow]:
        sql = f"""
            SELECT id, title, updated_at, article_count
            FROM {self._table}
            ORDER BY updated_at DESC, id DESC
        """
        with _cursor(self._resolve_conn(ctx)) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        return [
            IssueRow(
                id=int(row[0]),
                title=str(row[1]),
                updated_at=row[2],
                article_count=int(row[3]),
            )
            for row in rows
        ]

    def _resolve_conn(self, ctx: object) -> Any:
        return ctx if ctx is not None else self._conn


class PostgresArticleRepository:
    def __init__(self, conn: Any, table: str = "article") -> None:
        self._conn = conn
        self._table = table

    def create(
        self,
        ctx: object,
        title: str,
        content: str,
        source: str,
        url: str,
        published_at: datetime | None,
    ) -> int:
        title_hash = hashlib.sha256(title.strip().encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc)
        sql = f"""
            INSERT INTO {self._table} (
                issue_id, title, content, source, url, title_hash, created_at, published_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        with _cursor(self._resolve_conn(ctx)) as cur:
            cur.execute(sql, (0, title, content, source, url, title_hash, now, published_at))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("failed to create article")
        return int(row[0])

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
        table: str = "issue_embedding",
        similarity_limit: int = 20,
    ) -> None:
        self._conn = conn
        self._table = table
        self._similarity_limit = similarity_limit
        self._drop_sparse_columns_if_exists()

    def create(self, ctx: object, item: Any) -> None:
        sql = f"""
            INSERT INTO {self._table} (issue_id, dense, created_at)
            VALUES (%s, %s::vector, %s)
        """
        with _cursor(self._resolve_conn(ctx)) as cur:
            cur.execute(sql, (int(item.issue_id), _to_pgvector_literal(item.dense), datetime.now(timezone.utc)))

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

    def _drop_sparse_columns_if_exists(self) -> None:
        sql = f"""
            ALTER TABLE {self._table}
            DROP COLUMN IF EXISTS sparse_indice,
            DROP COLUMN IF EXISTS sparse_value
        """
        with _cursor(self._conn) as cur:
            cur.execute(sql)


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
