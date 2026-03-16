from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Reads all values from .env file."""

    # Redis
    redis_host: str
    redis_port: int
    redis_url: str

    # CORS
    cors_origins: List[str]

    # Chunking
    chunk_size: int
    chunk_overlap: int

    # Retrieval
    top_k_results: int

    # Model
    embedding_model: str
    llm_model: str
    llm_temperature: float
    llm_num_predict: int
    llm_num_ctx: int
    llm_num_thread: int
    
    # PDF Processing
    min_page_length: int
    
    # Paths
    vector_store_path: str
    collection_name: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()