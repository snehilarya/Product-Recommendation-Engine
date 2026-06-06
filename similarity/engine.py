# similarity/engine.py
import logging
from functools import lru_cache
from typing import Optional, List

import numpy as np
from similarity.config import settings

# Setup structured logging
logger = logging.getLogger(__name__)

# Module-level state
_id_to_index: Optional[dict] = None
_index_to_id: Optional[list] = None
_metadata: Optional[dict] = None  # Added to store prices for filtering
_features: Optional[np.ndarray] = None
_hnsw_index = None


def initialize(data_path: str) -> None:
    """Load data, build feature vectors, and build the HNSW index."""
    global _id_to_index, _index_to_id, _metadata, _features, _hnsw_index

    from similarity.data_loader import load_products
    from similarity.feature_builder import FeatureBuilder
    from similarity.index import SimilarityIndex

    logger.info(f"Loading products from {data_path}...")
    df, _id_to_index, _index_to_id = load_products(data_path)
    
    # Store price metadata for the Post-Filter
    _metadata = df.set_index("uniq_id")[["sales_price"]].to_dict("index")
    logger.info(f"Loaded {len(df)} products.")

    logger.info("Building feature vectors (TF-IDF + TruncatedSVD + numerics)...")
    builder = FeatureBuilder()
    _features = builder.build(df)
    logger.info(f"Feature matrix shape: {_features.shape}")

    logger.info("Building HNSW index...")
    _hnsw_index = SimilarityIndex(dim=_features.shape[1], max_elements=len(df))
    _hnsw_index.build(_features)
    logger.info("HNSW index ready for queries.")


def product_count() -> int:
    return len(_index_to_id) if _index_to_id is not None else 0


@lru_cache(maxsize=1000)
def find_similar_products(product_id: str, num_similar: int) -> List[str]:
    """
    Return a list of num_similar product IDs most similar to product_id,
    filtered dynamically to respect the user's price intent.
    """
    if _id_to_index is None:
        raise RuntimeError("Engine not initialized. Call initialize() first.")

    if product_id not in _id_to_index:
        raise KeyError(f"Product '{product_id}' not found in dataset")

    target_price = _metadata[product_id].get("sales_price")
    query_row_index = _id_to_index[product_id]
    query_vector = _features[query_row_index]

    # Ask the graph for a larger candidate pool to accommodate the filter
    candidate_pool_size = num_similar * 10
    neighbor_indices = _hnsw_index.query(
        vector=query_vector,
        k=candidate_pool_size,
        exclude_index=query_row_index
    )

    similar_ids = []
    for i in neighbor_indices:
        candidate_id = _index_to_id[i]
        candidate_price = _metadata[candidate_id].get("sales_price")
        
        # Apply the Hard Filter business logic
        if target_price and candidate_price and not np.isnan(candidate_price):
            lower_bound = target_price * settings.PRICE_TOLERANCE_LOWER
            upper_bound = target_price * settings.PRICE_TOLERANCE_UPPER
            if not (lower_bound <= candidate_price <= upper_bound):
                continue  # Skip items outside the price boundary

        similar_ids.append(candidate_id)
        if len(similar_ids) == num_similar:
            break

    return similar_ids