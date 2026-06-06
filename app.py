import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Query
import similarity.engine as engine
from similarity.config import settings

# Configure root logger for the FastAPI application
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
    logger.info("Shutting down recommendation engine...")

app = FastAPI(
    title="Product Similarity Search",
    description="Find similar Amazon fashion products using HNSW + TF-IDF with Price Guardrails",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {"status": "ok", "products_loaded": engine.product_count()}

@app.get("/find_similar_products", response_model=List[str])
def find_similar_products(
    product_id: str,
    num_similar: int = Query(default=5, ge=1, le=200)
) -> List[str]:
    try:
        return engine.find_similar_products(product_id, num_similar)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Product '{product_id}' not found in dataset"
        )