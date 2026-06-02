from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "local"
    log_level: str = "INFO"
    service_port: int = 8001

    model_name: str = "BAAI/bge-reranker-v2-m3"
    gemma_model_name: str = "BAAI/bge-reranker-v2-gemma"
    use_fp16: bool = True
    # Reranking is sequential over (query, doc) pairs — one in-flight call is enough.
    max_concurrent_inferences: int = 1
    # 0 => let torch decide
    torch_num_threads: int = 0
    max_docs_per_request: int = 128


settings = Settings()
