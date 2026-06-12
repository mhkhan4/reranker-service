import logging
import secrets
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status

from app import model as reranker
from app import model_gemma as gemma_reranker
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


def _verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> None:
    if settings.api_key and not secrets.compare_digest(x_api_key or "", settings.api_key):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")


@router.post("/rerank", response_model=RerankResponse, dependencies=[Depends(_verify_api_key)])
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


@router.post("/rerank/gemma", response_model=RerankResponse, dependencies=[Depends(_verify_api_key)])
async def rerank_gemma(req: RerankRequest) -> RerankResponse:
    if len(req.documents) > settings.max_docs_per_request:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Too many documents: max {settings.max_docs_per_request}, got {len(req.documents)}",
        )
    if not gemma_reranker.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemma model is still loading — retry after /ready returns 200",
        )

    logger.debug("Gemma reranking %d doc(s), top_n=%s", len(req.documents), req.top_n)
    try:
        ranked = await gemma_reranker.rerank(req.query, req.documents, req.top_n)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    return RerankResponse(
        model=settings.gemma_model_name,
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
