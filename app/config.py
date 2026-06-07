from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "local"
    log_level: str = "INFO"
    service_port: int = 8001

    model_name: str = "BAAI/bge-reranker-v2-m3"
    use_fp16: bool = True
    devices: str = "cpu"
    # Reranking is sequential over (query, doc) pairs — one in-flight call is enough.
    max_concurrent_inferences: int = 1
    # 0 => let torch decide; ignored when running on GPU
    torch_num_threads: int = 0
    max_docs_per_request: int = 128

    # Static API key for request authentication. Empty string disables auth (local dev).
    api_key: str = ""


settings = Settings()
