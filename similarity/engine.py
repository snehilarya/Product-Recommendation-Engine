from functools import lru_cache
from typing import Optional

import numpy as np

# Module-level state — set once during initialize(), read on every query
_id_to_index: Optional[dict] = None
_index_to_id: Optional[list] = None
_features: Optional[np.ndarray] = None
_hnsw_index = None


def initialize(data_path: str) -> None:
    """
    Load data, build feature vectors, and build the HNSW index.
    Must be called once before any calls to find_similar_products().
    """
    global _id_to_index, _index_to_id, _features, _hnsw_index

    from similarity.data_loader import load_products
    from similarity.feature_builder import FeatureBuilder
    from similarity.index import SimilarityIndex

    print(f"Loading products from {data_path}...")
    df, _id_to_index, _index_to_id = load_products(data_path)
    print(f"Loaded {len(df)} products.")

    print("Building feature vectors (TF-IDF + TruncatedSVD + numerics)...")
    builder = FeatureBuilder()
    _features = builder.build(df)
    print(f"Feature matrix shape: {_features.shape}")

    print("Building HNSW index...")
    _hnsw_index = SimilarityIndex(dim=_features.shape[1], max_elements=len(df))
    _hnsw_index.build(_features)
    print("HNSW index ready.")


def product_count() -> int:
    """Return number of products loaded. 0 if not yet initialized."""
    return len(_index_to_id) if _index_to_id is not None else 0


@lru_cache(maxsize=1000)
def find_similar_products(product_id: str, num_similar: int) -> list:
    """
    Return a list of num_similar product IDs most similar to product_id.

    Results are cached by (product_id, num_similar) — repeated identical
    queries return instantly without hitting the HNSW index.

    Raises:
        RuntimeError: if initialize() has not been called
        KeyError: if product_id is not in the loaded dataset
    """
    if _id_to_index is None:
        raise RuntimeError("Engine not initialized. Call initialize() first.")

    if product_id not in _id_to_index:
        raise KeyError(f"Product '{product_id}' not found in dataset")

    query_row_index = _id_to_index[product_id]
    query_vector = _features[query_row_index]

    neighbor_indices = _hnsw_index.query(
        vector=query_vector,
        k=num_similar,
        exclude_index=query_row_index
    )

    similar_ids = [_index_to_id[i] for i in neighbor_indices]
    return similar_ids
