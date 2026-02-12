CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS issue (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    article_count BIGINT NOT NULL DEFAULT 0,
    started_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS article (
    id BIGSERIAL PRIMARY KEY,
    issue_id BIGINT NOT NULL DEFAULT 0,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    url TEXT NOT NULL,
    title_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    published_at TIMESTAMPTZ NULL
);

CREATE INDEX IF NOT EXISTS idx_article_title_hash ON article (title_hash);
CREATE INDEX IF NOT EXISTS idx_article_issue_id ON article (issue_id);

CREATE TABLE IF NOT EXISTS article_embedding (
    article_id BIGINT PRIMARY KEY REFERENCES article(id) ON DELETE CASCADE,
    dense VECTOR(768) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS issue_embedding (
    issue_id BIGINT PRIMARY KEY REFERENCES issue(id) ON DELETE CASCADE,
    dense VECTOR(768),
    created_at TIMESTAMPTZ NOT NULL
);
