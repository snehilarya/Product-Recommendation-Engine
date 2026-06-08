import hashlib
import logging
import os
import pickle
import zipfile
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

# Cache file names written inside settings.CACHE_DIR
_CACHE_INDEX_FILE    = "hnsw_index.bin"
_CACHE_PIPELINE_FILE = "feature_pipeline.pkl"
_CACHE_META_FILE     = "meta.pkl"


def _config_hash() -> str:
    """
    Short hash of the ML hyperparameters that affect the built index.
    Used as the cache subdirectory name — if any param changes, the old
    cache is in a different directory and a fresh build is triggered automatically.
    """
    key = (
        f"v={settings.FEATURE_VERSION}"
        f"_svd={settings.SVD_COMPONENTS}"
        f"_w={settings.NUMERIC_WEIGHT}"
        f"_M={settings.HNSW_M}"
        f"_ef={settings.HNSW_EF_CONSTRUCTION}"
    )
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _cache_dir() -> str:
    """Return the versioned cache directory for the current config."""
    if not settings.CACHE_DIR:
        return ""
    return os.path.join(settings.CACHE_DIR, _config_hash())


def initialize(data_path: str) -> None:
    """
    Load data, build feature vectors, and build the HNSW index.
    Must be called once before any calls to find_similar_products().

    On the first run this takes ~2-3 seconds to build everything from scratch,
    then saves the result to CACHE_DIR. Subsequent restarts load from disk in
    ~100ms instead of rebuilding.

    Set CACHE_DIR='' to disable caching (always rebuild).
    """
    global _id_to_index, _index_to_id, _features, _prices, _hnsw_index

    from similarity.data_loader import load_products
    from similarity.feature_builder import FeatureBuilder
    from similarity.index import SimilarityIndex

    _ensure_data_file(data_path)

    if _try_load_from_cache():
        return

    logger.info(f"Loading products from {data_path}...")
    df, _id_to_index, _index_to_id = load_products(data_path)
    logger.info(f"Loaded {len(df)} products.")

    _prices = df["sales_price"].values.astype(float)

    logger.info("Building feature vectors (TF-IDF + TruncatedSVD + numerics)...")
    builder = FeatureBuilder()
    _features = builder.build(df)
    logger.info(f"Feature matrix shape: {_features.shape}")

    logger.info("Building HNSW index...")
    _hnsw_index = SimilarityIndex(dim=_features.shape[1], max_elements=len(df))
    _hnsw_index.build(_features)
    logger.info("HNSW index ready.")

    _save_to_cache(builder)


def _try_load_from_cache() -> bool:
    """
    Attempt to load a previously built index from the versioned cache dir.
    Returns True if successful, False if cache is absent or corrupt.
    If the config changed (different hash), the old cache dir won't exist
    and a fresh build is triggered automatically.
    """
    global _id_to_index, _index_to_id, _features, _prices, _hnsw_index

    cache_dir = _cache_dir()
    if not cache_dir:
        return False

    index_path    = os.path.join(cache_dir, _CACHE_INDEX_FILE)
    pipeline_path = os.path.join(cache_dir, _CACHE_PIPELINE_FILE)
    meta_path     = os.path.join(cache_dir, _CACHE_META_FILE)

    if not all(os.path.exists(p) for p in [index_path, pipeline_path, meta_path]):
        return False

    try:
        logger.info(f"Loading cached index from {cache_dir}...")
        from similarity.index import SimilarityIndex

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        _id_to_index = meta["id_to_index"]
        _index_to_id = meta["index_to_id"]
        _prices      = meta["prices"]
        _features    = meta["features"]

        _hnsw_index = SimilarityIndex.load(
            index_path,
            dim=_features.shape[1],
            max_elements=len(_index_to_id),
        )
        logger.info(f"Loaded {len(_index_to_id)} products from cache.")
        return True

    except Exception as e:
        logger.warning(f"Cache load failed ({e}), rebuilding from scratch.")
        return False


def _save_to_cache(builder) -> None:
    """Persist the built index and pipeline to the versioned cache dir."""
    cache_dir = _cache_dir()
    if not cache_dir:
        return

    os.makedirs(cache_dir, exist_ok=True)

    logger.info(f"Saving index cache to {cache_dir}...")
    _hnsw_index.save(os.path.join(cache_dir, _CACHE_INDEX_FILE))
    builder.save(os.path.join(cache_dir, _CACHE_PIPELINE_FILE))

    with open(os.path.join(cache_dir, _CACHE_META_FILE), "wb") as f:
        pickle.dump({
            "id_to_index": _id_to_index,
            "index_to_id": _index_to_id,
            "prices":      _prices,
            "features":    _features,
        }, f)

    logger.info("Index cache saved.")


def product_count() -> int:
    """Return number of products loaded. 0 if not yet initialized."""
    return len(_index_to_id) if _index_to_id is not None else 0


# Query result cache — bounded in-process dict, capped at 10k entries (FIFO eviction).
# Repeated identical queries return instantly without hitting the HNSW index.
_CACHE_MAX = 10_000
_cache: dict = {}


def _cache_get(key: str):
    return _cache.get(key)


def _cache_set(key: str, value):
    if len(_cache) >= _CACHE_MAX:
        del _cache[next(iter(_cache))]
    _cache[key] = value


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

    cache_key = f"{product_id}:{num_similar}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    result = _compute_similar(product_id, num_similar)
    _cache_set(cache_key, result)
    return result


def _compute_similar(product_id: str, num_similar: int) -> list:
    query_row_index = _id_to_index[product_id]
    query_vector = _features[query_row_index]
    query_price = _prices[query_row_index]

    candidates = _hnsw_index.query(
        vector=query_vector,
        k=num_similar * 5,
        exclude_index=query_row_index
    )

    filtered = _filter_by_price_band(candidates, query_price)

    if len(filtered) < num_similar:
        filtered = candidates

    return [_index_to_id[i] for i in filtered[:num_similar]]


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
