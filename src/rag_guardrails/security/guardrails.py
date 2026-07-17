"""Wraps generation in NeMo Guardrails' input/output self-check rails."""

from nemoguardrails import LLMRails, RailsConfig

from ..config import RAGConfig


class GuardrailsEngine:
    """Runs input/output rails (see config/rails/config.yml) around the LLM call."""

    def __init__(self, config: RAGConfig, llm):
        self.rails_config = RailsConfig.from_path(config.guardrails_config_path)
        self.rails = LLMRails(self.rails_config, llm=llm)

    async def generate(self, messages: list[dict]) -> str:
        response = await self.rails.generate_async(messages=messages)
        if isinstance(response, dict):
            return response.get("content", "")
        return response
