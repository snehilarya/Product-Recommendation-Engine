import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Query

import similarity.engine as engine

DATA_PATH = os.environ.get(
    "DATA_PATH",
    # default path for local development — override with DATA_PATH env var in Docker
    os.path.join(
        os.path.expanduser("~"),
        "CodeBase", "sap-cxii-tech-ex-01", "data",
        "marketing_sample_for_amazon_com-amazon_fashion_products.ldjson"
    )
)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Build all indexes at startup so the first request is fast."""
    engine.initialize(DATA_PATH)
    yield


app = FastAPI(
    title="Product Similarity Search",
    description="Find similar Amazon fashion products using HNSW + TF-IDF",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {"status": "ok", "products_loaded": engine.product_count()}


@app.get("/find_similar_products")
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
