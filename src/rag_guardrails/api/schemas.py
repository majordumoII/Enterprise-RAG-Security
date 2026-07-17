from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    user_id: str
    clearance: str = Field(
        default="public", description="public | internal | confidential | restricted"
    )
    roles: list[str] = Field(default_factory=list)
    top_k: int | None = None


class SourceChunk(BaseModel):
    filename: str
    chunk_index: int
    content: str
    similarity: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
