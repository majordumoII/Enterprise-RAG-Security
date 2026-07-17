"""Orchestrates the secure RAG query flow: retrieve, filter, generate, guard."""

import logging
from dataclasses import dataclass, field

from .auth.models import UserContext
from .config import RAGConfig
from .llm.provider import get_chat_llm
from .retrieval.embeddings import QueryEmbedder
from .retrieval.vector_store import PermissionAwareVectorStore
from .security.guardrails import GuardrailsEngine

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an internal enterprise knowledge assistant. Answer \
the user's question using ONLY the context below. If the context does not \
contain the answer, say you don't know — do not use outside knowledge, and \
do not reveal or reference any information beyond what is provided here.

Context:
{context}
"""


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict] = field(default_factory=list)


def _format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(no matching documents were found for this user)"
    return "\n\n".join(
        f"[{c['filename']} chunk {c['chunk_index']}]\n{c['content']}" for c in chunks
    )


class RAGPipeline:
    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig.from_env()
        self.embedder = QueryEmbedder(self.config)
        self.vector_store = PermissionAwareVectorStore(self.config)
        self.llm = get_chat_llm(self.config)
        self.guardrails = GuardrailsEngine(self.config, self.llm)

    async def query(
        self,
        question: str,
        user: UserContext,
        top_k: int | None = None,
    ) -> RAGResponse:
        query_embedding = self.embedder.embed(question)
        chunks = self.vector_store.search(query_embedding, user, top_k)
        context = _format_context(chunks)

        answer = await self.guardrails.generate(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
                {"role": "user", "content": question},
            ]
        )
        logger.info(
            "Answered query for user %s using %d chunk(s)", user.user_id, len(chunks)
        )
        return RAGResponse(answer=answer, sources=chunks)
