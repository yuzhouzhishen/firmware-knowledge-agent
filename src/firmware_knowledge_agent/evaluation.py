from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from firmware_knowledge_agent.models import (
    EvaluationQuestion,
    EvaluationReport,
    EvaluationResult,
    EvaluationRunConfig,
    KnowledgeChunk,
    SearchHit,
)
from firmware_knowledge_agent.service import FirmwareKnowledgeService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = PROJECT_ROOT / "data/sample/sources.json"
DEFAULT_QUESTIONS = PROJECT_ROOT / "data/eval/baseline.json"


def evaluate(
    service: FirmwareKnowledgeService,
    questions: list[EvaluationQuestion],
    *,
    top_k: int = 3,
) -> EvaluationReport:
    results: list[EvaluationResult] = []
    for question in questions:
        response = service.search(question.question, top_k=top_k)
        retrieved = [hit.chunk.source_id for hit in response.hits]
        retrieved_chunk_ids = [hit.chunk.id for hit in response.hits]
        abstained = response.status == "no_evidence"
        first_relevant_rank = _first_relevant_rank(
            response.hits,
            question,
        )
        hit = first_relevant_rank is not None
        passed = (
            hit
            if question.expected_answerable
            else abstained
        )
        results.append(
            EvaluationResult(
                question_id=question.id,
                expected_answerable=question.expected_answerable,
                passed=passed,
                hit=hit,
                abstained=abstained,
                reciprocal_rank=(
                    1.0 / first_relevant_rank
                    if first_relevant_rank is not None
                    else 0.0
                ),
                retrieved_source_ids=retrieved,
                retrieved_chunk_ids=retrieved_chunk_ids,
            )
        )

    question_count = len(results)
    if question_count == 0:
        raise ValueError("evaluation requires at least one question")
    answerable_results = [
        result for result in results if result.expected_answerable
    ]
    unanswerable_results = [
        result for result in results if not result.expected_answerable
    ]
    answerable_count = len(answerable_results)
    unanswerable_count = len(unanswerable_results)
    return EvaluationReport(
        top_k=top_k,
        question_count=question_count,
        answerable_count=answerable_count,
        unanswerable_count=unanswerable_count,
        hit_at_k=round(
            (
                sum(result.hit for result in answerable_results)
                / answerable_count
            )
            if answerable_count
            else 0.0,
            4,
        ),
        mrr=round(
            (
                sum(
                    result.reciprocal_rank
                    for result in answerable_results
                )
                / answerable_count
            )
            if answerable_count
            else 0.0,
            4,
        ),
        abstention_accuracy=(
            round(
                sum(result.passed for result in unanswerable_results)
                / unanswerable_count,
                4,
            )
            if unanswerable_count
            else None
        ),
        overall_accuracy=round(
            sum(result.passed for result in results) / question_count,
            4,
        ),
        results=results,
    )


def _first_relevant_rank(
    hits: list[SearchHit],
    question: EvaluationQuestion,
) -> int | None:
    if not question.expected_answerable:
        return None
    expected = set(question.expected_source_ids)
    return next(
        (
            index
            for index, hit in enumerate(hits, start=1)
            if (
                hit.chunk.source_id in expected
                and _contains_expected_evidence(
                    "\n".join(
                        (
                            hit.chunk.title,
                            hit.chunk.section,
                            hit.chunk.content,
                        )
                    ),
                    question.expected_content_term_groups,
                )
            )
        ),
        None,
    )


def _contains_expected_evidence(
    content: str,
    term_groups: list[list[str]],
) -> bool:
    normalized_content = content.casefold()
    return all(
        any(term.casefold() in normalized_content for term in group)
        for group in term_groups
    )


def load_questions(path: Path) -> list[EvaluationQuestion]:
    raw_questions = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvaluationQuestion.model_validate(question)
        for question in raw_questions
    ]


def audit_questions(
    questions: list[EvaluationQuestion],
    chunks: list[KnowledgeChunk],
    *,
    require_same_chunk: bool = False,
) -> None:
    question_ids = [question.id for question in questions]
    duplicate_ids = sorted(
        question_id
        for question_id in set(question_ids)
        if question_ids.count(question_id) > 1
    )
    if duplicate_ids:
        raise ValueError(
            "duplicate evaluation question ids: "
            + ", ".join(duplicate_ids)
        )

    chunks_by_source: dict[str, list[KnowledgeChunk]] = {}
    for chunk in chunks:
        chunks_by_source.setdefault(chunk.source_id, []).append(chunk)

    errors: list[str] = []
    for question in questions:
        if not question.expected_answerable:
            continue
        missing_sources = sorted(
            set(question.expected_source_ids) - chunks_by_source.keys()
        )
        if missing_sources:
            errors.append(
                f"{question.id}: missing sources {', '.join(missing_sources)}"
            )
            continue
        if not question.expected_content_term_groups:
            continue
        candidate_chunks = [
            chunk
            for source_id in question.expected_source_ids
            for chunk in chunks_by_source[source_id]
        ]
        candidate_texts = [
            "\n".join((chunk.title, chunk.section, chunk.content))
            for chunk in candidate_chunks
        ]
        evidence_is_reachable = (
            any(
                _contains_expected_evidence(
                    text,
                    question.expected_content_term_groups,
                )
                for text in candidate_texts
            )
            if require_same_chunk
            else _contains_expected_evidence(
                "\n".join(candidate_texts),
                question.expected_content_term_groups,
            )
        )
        if not evidence_is_reachable:
            errors.append(
                (
                    f"{question.id}: no expected chunk contains all evidence "
                    "groups"
                    if require_same_chunk
                    else (
                        f"{question.id}: expected sources do not contain all "
                        "evidence groups"
                    )
                )
            )
    if errors:
        raise ValueError(
            "evaluation label audit failed:\n" + "\n".join(errors)
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Evaluate firmware document retrieval.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--retriever",
        choices=("bm25", "vector", "hybrid"),
        default=os.getenv("FIRMWARE_RAG_RETRIEVER", "bm25"),
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path(
            os.getenv(
                "FIRMWARE_RAG_EMBEDDING_CACHE",
                "var/embeddings.json",
            )
        ),
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
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON evaluation report.",
    )
    parser.add_argument(
        "--strict-label-audit",
        action="store_true",
        help=(
            "Require every answerable label to be satisfiable by one chunk "
            "before running retrieval."
        ),
    )
    args = parser.parse_args()

    service = FirmwareKnowledgeService(
        args.catalog,
        retriever_mode=args.retriever,
        embedding_cache_path=args.embedding_cache,
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
    questions = load_questions(args.questions)
    try:
        audit_questions(
            questions,
            service.chunks,
            require_same_chunk=args.strict_label_audit,
        )
        report = evaluate(
            service,
            questions,
            top_k=args.top_k,
        )
        report.run_config = EvaluationRunConfig(
            retriever=args.retriever,
            reranker=args.reranker,
            embedding_provider=os.getenv(
                "FIRMWARE_RAG_EMBEDDING_PROVIDER",
                "gemini",
            ),
            embedding_model=os.getenv(
                "FIRMWARE_RAG_EMBEDDING_MODEL",
                (
                    "qwen3-embedding:0.6b"
                    if os.getenv(
                        "FIRMWARE_RAG_EMBEDDING_PROVIDER",
                        "gemini",
                    )
                    == "ollama"
                    else "models/gemini-embedding-001"
                ),
            ),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            vector_min_score=args.vector_min_score,
            strict_label_audit=args.strict_label_audit,
            catalog_sha256=_sha256(args.catalog),
            questions_sha256=_sha256(args.questions),
        )
    finally:
        service.close()
    report_json = report.model_dump_json(indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{report_json}\n", encoding="utf-8")
    print(report_json)


if __name__ == "__main__":
    main()
