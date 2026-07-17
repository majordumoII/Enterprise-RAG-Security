from src.rag_guardrails.config import RAGConfig


class TestRAGConfig:
    def test_from_env_reads_values(self, config: RAGConfig):
        assert config.project_id == "test-project"
        assert config.vector_table == "test_chunks"
        assert config.llm_provider == "openai"
        assert config.top_k == 5

    def test_defaults_when_unset(self, monkeypatch):
        for var in [
            "LLM_PROVIDER",
            "OPENAI_MODEL",
            "GUARDRAILS_CONFIG_PATH",
            "TOP_K",
            "VECTOR_TABLE",
        ]:
            monkeypatch.delenv(var, raising=False)

        cfg = RAGConfig.from_env()

        assert cfg.llm_provider == "openai"
        assert cfg.openai_model == "gpt-4.1-mini"
        assert cfg.guardrails_config_path == "config/rails"
        assert cfg.top_k == 5
        assert cfg.vector_table == "document_chunks"
