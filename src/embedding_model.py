from pydantic import BaseModel


class GeneratedEmbedding(BaseModel):
    dense: list[float]


class ClusterIssueEmbedding(BaseModel):
    issue_id: int
    dense: list[float]
