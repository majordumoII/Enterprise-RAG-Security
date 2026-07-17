"""Pluggable generation LLM: OpenAI or DeepSeek (OpenAI-compatible API).

One LangChain ChatOpenAI-based client serves both providers — DeepSeek's
chat completions endpoint is OpenAI-compatible, so only api_key/base_url/model
need to change. The returned object also gets passed to GuardrailsEngine so
the same model handles both generation and NeMo Guardrails' self-check rails.
"""

from langchain_openai import ChatOpenAI

from ..config import RAGConfig


def get_chat_llm(config: RAGConfig) -> ChatOpenAI:
    if config.llm_provider == "openai":
        return ChatOpenAI(api_key=config.openai_api_key, model=config.openai_model)
    if config.llm_provider == "deepseek":
        return ChatOpenAI(
            api_key=config.deepseek_api_key,
            base_url=config.deepseek_base_url,
            model=config.deepseek_model,
        )
    raise ValueError(
        f"Unknown LLM_PROVIDER {config.llm_provider!r}; expected 'openai' or 'deepseek'"
    )
