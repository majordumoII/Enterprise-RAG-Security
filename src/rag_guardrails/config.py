import os
from dataclasses import dataclass, field


@dataclass
class RAGConfig:
    # GCP / Vertex AI (query-time embeddings — must match Project 1's ingestion model/dims)
    project_id: str = field(
        default_factory=lambda: os.getenv("GOOGLE_CLOUD_PROJECT", "")
    )
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-005")
    )
    embedding_dimensions: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
    )
    embedding_location: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_LOCATION", "us-east1")
    )

    # Cloud SQL pgvector — same instance/table Project 1 populated
    pg_connection_string: str = field(
        default_factory=lambda: os.getenv("PG_CONNECTION_STRING", "")
    )
    vector_table: str = field(
        default_factory=lambda: os.getenv("VECTOR_TABLE", "document_chunks")
    )
    db_instance_connection_name: str = field(
        default_factory=lambda: os.getenv("DB_INSTANCE_CONNECTION_NAME", "")
    )
    db_name: str = field(default_factory=lambda: os.getenv("DB_NAME", "docpipeline"))
    db_user: str = field(default_factory=lambda: os.getenv("DB_USER", "pipeline"))
    db_password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))

    # Generation LLM (pluggable)
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "openai")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    )
    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
    )
    deepseek_model: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    )
    deepseek_base_url: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )

    # Guardrails
    guardrails_config_path: str = field(
        default_factory=lambda: os.getenv("GUARDRAILS_CONFIG_PATH", "config/rails")
    )

    # Retrieval
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))

    @classmethod
    def from_env(cls) -> "RAGConfig":
        return cls()
