import logging

from fastapi import APIRouter, HTTPException, status

from app import model as reranker
from app.config import settings
from app.schemas import (
    HealthResponse,
    ReadyResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest) -> RerankResponse:
    if len(req.documents) > settings.max_docs_per_request:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Too many documents: max {settings.max_docs_per_request}, got {len(req.documents)}",
        )
    if not reranker.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still loading — retry after /ready returns 200",
        )

    logger.debug("Reranking %d doc(s), top_n=%s", len(req.documents), req.top_n)
    try:
        ranked = await reranker.rerank(req.query, req.documents, req.top_n)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    return RerankResponse(
        model=settings.model_name,
        results=[RerankResult(index=idx, relevance_score=score) for idx, score in ranked],
    )



@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=reranker.is_loaded())


@router.get("/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    if not reranker.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not yet loaded",
        )
    return ReadyResponse(status="ready")
