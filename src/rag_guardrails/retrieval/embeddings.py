"""Query embedding via Vertex AI — must match Project 1's ingestion model/dims."""

import logging
from typing import Any

from google.cloud import aiplatform

from ..config import RAGConfig

logger = logging.getLogger(__name__)


class QueryEmbedder:
    """Embeds user queries into the same vector space as Project 1's stored chunks."""

    def __init__(self, config: RAGConfig):
        self.config = config
        aiplatform.init(project=config.project_id, location=config.embedding_location)
        self._model = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from vertexai.language_models import TextEmbeddingModel

            self._model = TextEmbeddingModel.from_pretrained(
                self.config.embedding_model_name
            )
        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed a single query string and return the vector."""
        embeddings = self.model.get_embeddings([text])
        return embeddings[0].values  # type: ignore[return-value]
