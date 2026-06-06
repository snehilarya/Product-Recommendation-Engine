# similarity/config.py
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo-relative path to the LDJSON file produced by unzipping data/archive.zip
_DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "marketing_sample_for_amazon_com-amazon_fashion_products.ldjson"
)

_DEFAULT_ZIP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "archive.zip"
)


class Settings(BaseSettings):
    # App Config — override with DATA_PATH env var in Docker/Kubernetes
    DATA_PATH: str = _DEFAULT_DATA_PATH
    ZIP_PATH: str = _DEFAULT_ZIP_PATH
    # Cache config
    # Directory where the built index and pipeline are cached between restarts.
    # Set to an empty string to disable caching (always rebuild from scratch).
    CACHE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".index_cache")
    # Redis URL for query result caching across restarts and replicas.
    # Leave empty to use the in-process LRU cache only (default, no Redis needed).
    REDIS_URL: str = ""

    # ML Pipeline Config
    # SVD_COMPONENTS=75 captures 34% text variance vs 29% at 50; marginal gain
    # flattens beyond 75 (measured on this dataset). See docs/tuning.md.
    SVD_COMPONENTS: int = 75
    # NUMERIC_WEIGHT scales [price, rating] dims relative to L2-normed text dims.
    # 0.3 gives same-category hit rate of 80.2% vs 75.3% at w=1.0 (measured).
    NUMERIC_WEIGHT: float = 0.3
    HNSW_M: int = 16
    HNSW_EF_CONSTRUCTION: int = 200
    HNSW_EF_QUERY: int = 50

    # Price band filter — candidates outside this range of the query price are dropped.
    # e.g. 0.33/3.0 means a $500 watch only returns products between ~$167 and $1500.
    PRICE_TOLERANCE_LOWER: float = 0.33
    PRICE_TOLERANCE_UPPER: float = 3.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
