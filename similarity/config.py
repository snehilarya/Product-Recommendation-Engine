# similarity/config.py
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Config — override with DATA_PATH env var in Docker/Kubernetes
    DATA_PATH: str = os.path.join(
        os.path.expanduser("~"),
        "CodeBase", "sap-cxii-tech-ex-01", "data",
        "marketing_sample_for_amazon_com-amazon_fashion_products.ldjson"
    )

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
