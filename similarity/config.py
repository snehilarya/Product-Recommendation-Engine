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

    # ML Pipeline Config
    SVD_COMPONENTS: int = 50
    HNSW_M: int = 16
    HNSW_EF_CONSTRUCTION: int = 200
    HNSW_EF_QUERY: int = 50

    # Price band filter — candidates outside this range of the query price are dropped.
    # e.g. 0.33/3.0 means a $500 watch only returns products between ~$167 and $1500.
    PRICE_TOLERANCE_LOWER: float = 0.33
    PRICE_TOLERANCE_UPPER: float = 3.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
