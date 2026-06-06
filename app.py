import logging
import threading
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

import similarity.engine as engine
from similarity.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Build all indexes at startup so the first request is fast."""
    logger.info("Initializing recommendation engine...")
    engine.initialize(settings.DATA_PATH)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Product Similarity Search",
    description="Find similar Amazon fashion products using HNSW + TF-IDF",
    lifespan=lifespan
)

# Semaphore-based concurrency cap: at most MAX_CONCURRENT requests are allowed
# to run find_similar_products at the same time. Requests beyond this cap get a
# 503 immediately instead of queuing indefinitely and exhausting memory.
_MAX_CONCURRENT = 50
_semaphore = threading.Semaphore(_MAX_CONCURRENT)


@app.middleware("http")
async def concurrency_limit(request: Request, call_next):
    if request.url.path == "/find_similar_products":
        acquired = _semaphore.acquire(blocking=False)
        if not acquired:
            return JSONResponse(
                status_code=503,
                content={"detail": "Server busy, please retry shortly."}
            )
        try:
            return await call_next(request)
        finally:
            _semaphore.release()
    return await call_next(request)


@app.get("/health")
def health():
    return {"status": "ok", "products_loaded": engine.product_count()}


@app.get("/find_similar_products", response_model=List[str])
def find_similar_products(
    product_id: str,
    num_similar: int = Query(default=5, ge=1, le=200)
) -> List[str]:
    """
    Return the num_similar most similar products to the given product_id.

    - product_id: uniq_id from the dataset
    - num_similar: how many similar products to return (1–200)
    """
    try:
        return engine.find_similar_products(product_id, num_similar)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product_id}' not found in dataset"
        )
