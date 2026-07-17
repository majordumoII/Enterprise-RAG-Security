"""FastAPI service exposing the secure RAG pipeline over HTTP.

`user_id`/`clearance`/`roles` are accepted directly on the request body as a
demo-level auth stand-in — swap for real SSO/JWT-derived identity before any
production or Project 3 integration.
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from ..auth.models import ClearanceLevel, UserContext
from ..config import RAGConfig
from ..pipeline import RAGPipeline
from .dependencies import get_pipeline
from .schemas import QueryRequest, QueryResponse, SourceChunk


def create_app(config: RAGConfig | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = RAGPipeline(config)
        yield

    app = FastAPI(title="Enterprise RAG Security Guardrails", lifespan=lifespan)

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    async def query(
        body: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)
    ) -> QueryResponse:
        user = UserContext(
            user_id=body.user_id,
            clearance=ClearanceLevel.from_str(body.clearance),
            roles=body.roles,
        )
        result = await pipeline.query(body.question, user, top_k=body.top_k)
        return QueryResponse(
            answer=result.answer,
            sources=[
                SourceChunk(
                    filename=c["filename"],
                    chunk_index=c["chunk_index"],
                    content=c["content"],
                    similarity=c["similarity"],
                )
                for c in result.sources
            ],
        )

    return app


app = create_app()
