# similarity/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App Config
    DATA_PATH: str = "data/marketing_sample_for_amazon_com-amazon_fashion_products.ldjson"
    
    # ML Pipeline Config
    SVD_COMPONENTS: int = 50
    HNSW_M: int = 16
    HNSW_EF_CONSTRUCTION: int = 200
    HNSW_EF_QUERY: int = 50
    
    # Business Logic Config (The Hard Filter)
    PRICE_TOLERANCE_LOWER: float = 0.5
    PRICE_TOLERANCE_UPPER: float = 2.0

    # Modern Pydantic v2 Config
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()