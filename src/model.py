from pydantic import BaseModel
import datetime

class Article(BaseModel):
    title: str
    content: str


class IssueEmbedding(BaseModel):
    dense: list[float]


class MatchCandidate(BaseModel):
    issue_id: int
    title: str
    score: float
    similarity: float

class Source(BaseModel):
    url: str
    reference: str
    category: str


class CrawlItem(BaseModel):
    title: str
    content: str
    source: Source
    url: str
    published_at: datetime.datetime
