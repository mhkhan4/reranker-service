from pydantic import BaseModel, Field


class RerankRequest(BaseModel):
    query: str
    documents: list[str] = Field(min_length=1)
    top_n: int | None = None  # None => return all, sorted by score


class RerankResult(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    model: str
    results: list[RerankResult]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ReadyResponse(BaseModel):
    status: str
