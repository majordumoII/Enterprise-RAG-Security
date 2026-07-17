"""Entry point: query the secure RAG pipeline, tag permissions, or serve the API."""

import argparse
import asyncio
import logging

from dotenv import load_dotenv

# Load .env before anything else
load_dotenv()

from src.rag_guardrails.auth.models import ClearanceLevel, UserContext
from src.rag_guardrails.config import RAGConfig
from src.rag_guardrails.pipeline import RAGPipeline
from src.rag_guardrails.retrieval.vector_store import PermissionAwareVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Secure RAG query layer over Project 1's pgvector store"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Query the pipeline
    p_query = sub.add_parser("query", help="Ask a question through the secure RAG pipeline")
    p_query.add_argument("question", help="Natural language question")
    p_query.add_argument("--user-id", required=True)
    p_query.add_argument(
        "--clearance",
        default="public",
        help="public | internal | confidential | restricted (default: public)",
    )
    p_query.add_argument("--roles", default="", help="Comma-separated role list")
    p_query.add_argument("--top-k", type=int, default=None)

    # Tag permissions onto already-ingested chunks (demo/seeding utility)
    p_tag = sub.add_parser(
        "tag-permissions", help="Tag Project 1's ingested chunks with ACL metadata"
    )
    p_tag.add_argument("filename", help="filename column value to match (as stored by Project 1)")
    p_tag.add_argument(
        "--sensitivity", default=None, help="public | internal | confidential | restricted"
    )
    p_tag.add_argument("--roles", default=None, help="Comma-separated allowed roles")

    # Serve the FastAPI app
    p_serve = sub.add_parser("serve", help="Run the FastAPI query service")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)

    # Config info
    sub.add_parser("info", help="Show pipeline configuration")

    args = parser.parse_args()
    config = RAGConfig.from_env()

    if args.command == "info":
        print("RAG Configuration:")
        for key, val in config.__dict__.items():
            print(f"  {key}: {val}")
        return

    if args.command == "query":
        roles = [r.strip() for r in args.roles.split(",") if r.strip()]
        user = UserContext(
            user_id=args.user_id,
            clearance=ClearanceLevel.from_str(args.clearance),
            roles=roles,
        )
        pipeline = RAGPipeline(config)
        result = asyncio.run(pipeline.query(args.question, user, top_k=args.top_k))
        pipeline.vector_store.close()

        print(f"\n{result.answer}\n")
        print(f"Sources ({len(result.sources)}):")
        for s in result.sources:
            print(f"  [{s['similarity']:.3f}] {s['filename']}#chunk-{s['chunk_index']}")
        return

    if args.command == "tag-permissions":
        roles = [r.strip() for r in args.roles.split(",") if r.strip()] if args.roles else None
        store = PermissionAwareVectorStore(config)
        updated = store.update_permissions(
            filename=args.filename, sensitivity=args.sensitivity, allowed_roles=roles
        )
        store.close()
        print(f"Tagged {updated} chunk(s) for {args.filename}")
        return

    if args.command == "serve":
        import uvicorn

        from src.rag_guardrails.api.app import app

        uvicorn.run(app, host=args.host, port=args.port)
        return


if __name__ == "__main__":
    main()
