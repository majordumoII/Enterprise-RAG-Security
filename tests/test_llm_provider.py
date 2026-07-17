import pytest

from src.rag_guardrails.config import RAGConfig
from src.rag_guardrails.llm.provider import get_chat_llm


class TestGetChatLlm:
    def test_openai_provider(self, config: RAGConfig, mocker):
        mock_chat_openai = mocker.patch("src.rag_guardrails.llm.provider.ChatOpenAI")

        get_chat_llm(config)

        mock_chat_openai.assert_called_once_with(
            api_key=config.openai_api_key, model=config.openai_model
        )

    def test_deepseek_provider(self, config: RAGConfig, mocker):
        config.llm_provider = "deepseek"
        mock_chat_openai = mocker.patch("src.rag_guardrails.llm.provider.ChatOpenAI")

        get_chat_llm(config)

        mock_chat_openai.assert_called_once_with(
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            model=config.deepseek_model,
        )

    def test_unknown_provider_raises(self, config: RAGConfig):
        config.llm_provider = "anthropic"
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            get_chat_llm(config)
