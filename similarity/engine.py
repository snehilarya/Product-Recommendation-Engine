import logging
import os
import zipfile
from functools import lru_cache
from typing import Optional

import numpy as np

from similarity.config import settings

logger = logging.getLogger(__name__)

# Module-level state — set once during initialize(), read on every query
_id_to_index: Optional[dict] = None
_index_to_id: Optional[list] = None
_features: Optional[np.ndarray] = None
_prices: Optional[np.ndarray] = None   # sales_price per row index, NaN if missing
_hnsw_index = None


def initialize(data_path: str) -> None:
    """
    Load data, build feature vectors, and build the HNSW index.
    Must be called once before any calls to find_similar_products().
    """
    global _id_to_index, _index_to_id, _features, _prices, _hnsw_index

    from similarity.data_loader import load_products
    from similarity.feature_builder import FeatureBuilder
    from similarity.index import SimilarityIndex

    _ensure_data_file(data_path)

    logger.info(f"Loading products from {data_path}...")
    df, _id_to_index, _index_to_id = load_products(data_path)
    logger.info(f"Loaded {len(df)} products.")

    # Keep a price array for post-retrieval price band filtering.
    # NaN means price is unknown — those products are never filtered out.
    _prices = df["sales_price"].values.astype(float)

    logger.info("Building feature vectors (TF-IDF + TruncatedSVD + numerics)...")
    builder = FeatureBuilder()
    _features = builder.build(df)
    logger.info(f"Feature matrix shape: {_features.shape}")

    logger.info("Building HNSW index...")
    _hnsw_index = SimilarityIndex(dim=_features.shape[1], max_elements=len(df))
    _hnsw_index.build(_features)
    logger.info("HNSW index ready.")


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
    query_price = _prices[query_row_index]

    # Fetch more candidates than needed so we still have num_similar results
    # after the price band filter removes cross-segment noise.
    candidates = _hnsw_index.query(
        vector=query_vector,
        k=num_similar * 5,
        exclude_index=query_row_index
    )

    filtered = _filter_by_price_band(candidates, query_price)

    # Fall back to unfiltered results if price is unknown or too few survive
    if len(filtered) < num_similar:
        filtered = candidates

    similar_ids = [_index_to_id[i] for i in filtered[:num_similar]]
    return similar_ids


def _ensure_data_file(data_path: str) -> None:
    """
    If the LDJSON file doesn't exist yet, unzip it from archive.zip.
    This lets us ship the compressed dataset in the repo without the
    71 MB uncompressed file.
    """
    if os.path.exists(data_path):
        return

    zip_path = settings.ZIP_PATH
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Data file not found at {data_path} and no archive at {zip_path}. "
            f"Set the DATA_PATH environment variable to point to the dataset."
        )

    logger.info(f"Unzipping dataset from {zip_path}...")
    data_dir = os.path.dirname(data_path)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(data_dir)
        # The zip may contain a file with a different name (e.g. with a date suffix).
        # If the expected file still doesn't exist, rename whatever was extracted.
        if not os.path.exists(data_path):
            extracted = [
                os.path.join(data_dir, name)
                for name in archive.namelist()
                if name.endswith(".ldjson")
            ]
            if extracted:
                os.rename(extracted[0], data_path)
    logger.info("Dataset ready.")


def _filter_by_price_band(candidate_indices: list, query_price: float) -> list:
    """
    Remove candidates whose price is more than the configured tolerance
    higher or lower than the query product's price.

    A $500 watch with PRICE_TOLERANCE_UPPER=3.0 keeps results in the
    ~$167–$1500 range, preventing a $15 plastic watch from ranking above
    a $450 leather one just because they share the words 'black' and 'watch'.

    Products with unknown price (NaN) are always kept.
    If the query product itself has no price, filtering is skipped entirely.
    """
    if np.isnan(query_price):
        return candidate_indices

    result = []
    for row_index in candidate_indices:
        candidate_price = _prices[row_index]
        if np.isnan(candidate_price):
            result.append(row_index)
            continue
        ratio = candidate_price / query_price
        if settings.PRICE_TOLERANCE_LOWER <= ratio <= settings.PRICE_TOLERANCE_UPPER:
            result.append(row_index)
    return result
