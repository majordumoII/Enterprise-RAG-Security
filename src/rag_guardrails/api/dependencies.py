from fastapi import Request

from ..pipeline import RAGPipeline


def get_pipeline(request: Request) -> RAGPipeline:
    return request.app.state.pipeline
