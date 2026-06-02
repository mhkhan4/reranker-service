import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import model as reranker
from app import model_gemma as gemma_reranker
from app.api.routes import router
from app.config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "reranker-service starting — model=%s fp16=%s",
        settings.model_name,
        settings.use_fp16,
    )
    await asyncio.to_thread(reranker.load_model)
    await asyncio.to_thread(gemma_reranker.load_model)
    logger.info("reranker-service ready")
    yield
    await asyncio.to_thread(reranker.unload_model)
    await asyncio.to_thread(gemma_reranker.unload_model)
    logger.info("reranker-service shut down")


app = FastAPI(title="reranker-service", lifespan=lifespan)
app.include_router(router)
