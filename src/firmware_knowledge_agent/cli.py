from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from firmware_knowledge_agent.agentic_workflow import (
    AgenticRagService,
    build_answer_generator_from_env,
)
from firmware_knowledge_agent.evaluation import DEFAULT_CATALOG
from firmware_knowledge_agent.service import FirmwareKnowledgeService


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Search the firmware documentation corpus.",
    )
    parser.add_argument(
        "mode",
        choices=("search", "answer", "agent-answer"),
    )
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
    )
    parser.add_argument(
        "--retriever",
        choices=("bm25", "vector", "hybrid"),
        default=os.getenv("FIRMWARE_RAG_RETRIEVER", "bm25"),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("FIRMWARE_RAG_CHUNK_SIZE", "700")),
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("FIRMWARE_RAG_CHUNK_OVERLAP", "100")),
    )
    parser.add_argument(
        "--vector-min-score",
        type=float,
        default=float(
            os.getenv("FIRMWARE_RAG_VECTOR_MIN_SCORE", "0.55")
        ),
    )
    parser.add_argument(
        "--vector-store",
        type=Path,
        default=Path(
            os.getenv(
                "FIRMWARE_RAG_VECTOR_STORE",
                "var/vector-store",
            )
        ),
    )
    parser.add_argument(
        "--vector-collection",
        default=os.getenv(
            "FIRMWARE_RAG_VECTOR_COLLECTION",
            "firmware_knowledge",
        ),
    )
    parser.add_argument(
        "--reranker",
        choices=("none", "cross-encoder"),
        default=os.getenv("FIRMWARE_RAG_RERANKER", "none"),
    )
    parser.add_argument(
        "--reranker-model",
        default=os.getenv(
            "FIRMWARE_RAG_RERANKER_MODEL",
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        ),
    )
    parser.add_argument(
        "--reranker-model-file",
        default=os.getenv(
            "FIRMWARE_RAG_RERANKER_ONNX_FILE",
            "onnx/model_quint8_avx2.onnx",
        ),
    )
    parser.add_argument(
        "--reranker-cache",
        type=Path,
        default=Path(
            os.getenv(
                "FIRMWARE_RAG_RERANKER_CACHE",
                "var/models",
            )
        ),
    )
    parser.add_argument(
        "--reranker-candidates",
        type=int,
        default=int(
            os.getenv("FIRMWARE_RAG_RERANKER_CANDIDATES", "12")
        ),
    )
    args = parser.parse_args()

    service = FirmwareKnowledgeService(
        args.catalog,
        retriever_mode=args.retriever,
        embedding_cache_path=Path(
            os.getenv(
                "FIRMWARE_RAG_EMBEDDING_CACHE",
                "var/embeddings.json",
            )
        ),
        vector_store_path=args.vector_store,
        vector_collection=args.vector_collection,
        reranker_mode=args.reranker,
        reranker_model=args.reranker_model,
        reranker_model_file=args.reranker_model_file,
        reranker_cache_dir=args.reranker_cache,
        reranker_candidate_count=args.reranker_candidates,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        vector_min_score=args.vector_min_score,
    )
    if args.mode == "search":
        result = service.search(args.query, top_k=args.top_k)
    elif args.mode == "answer":
        result = service.answer(args.query, top_k=args.top_k)
    else:
        result = AgenticRagService(
            service,
            build_answer_generator_from_env(),
        ).answer(args.query, top_k=args.top_k)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
