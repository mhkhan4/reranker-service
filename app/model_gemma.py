import asyncio
import logging

import torch
from FlagEmbedding import FlagLLMReranker

from app.config import settings

logger = logging.getLogger(__name__)

_model: FlagLLMReranker | None = None
_infer_semaphore: asyncio.Semaphore | None = None


def load_model() -> None:
    global _model, _infer_semaphore
    if settings.torch_num_threads > 0:
        torch.set_num_threads(settings.torch_num_threads)
    logger.info("Loading %s (fp16=%s) on CPU…", settings.gemma_model_name, settings.use_fp16)
    _model = FlagLLMReranker(
        settings.gemma_model_name,
        use_fp16=settings.use_fp16,
        devices=settings.devices,
    )
    _infer_semaphore = asyncio.Semaphore(settings.max_concurrent_inferences)
    logger.info("Gemma reranker loaded.")


def unload_model() -> None:
    global _model
    _model = None


def is_loaded() -> bool:
    return _model is not None


async def rerank(
    query: str,
    documents: list[str],
    top_n: int | None,
) -> list[tuple[int, float]]:
    m = _model
    if m is None or _infer_semaphore is None:
        raise RuntimeError("Gemma reranker not loaded — service is still warming up")

    pairs = [[query, doc] for doc in documents]

    def _run() -> list[float]:
        result = m.compute_score(pairs, normalize=True)
        return result if isinstance(result, list) else [result]

    async with _infer_semaphore:
        scores: list[float] = await asyncio.to_thread(_run)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    if top_n is not None:
        ranked = ranked[:top_n]
    return ranked
