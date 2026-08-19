from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import re
from pathlib import Path

from dotenv import load_dotenv

from firmware_knowledge_agent.agentic_workflow import (
    AgenticRagService,
    build_answer_generator_from_env,
)
from firmware_knowledge_agent.evaluation import (
    DEFAULT_CATALOG,
    DEFAULT_QUESTIONS,
    _contains_expected_evidence,
    load_questions,
)
from firmware_knowledge_agent.models import (
    AnswerEvaluationReport,
    AnswerEvaluationResult,
    EvaluationQuestion,
)
from firmware_knowledge_agent.service import (
    build_knowledge_service_from_env,
)


def evaluate_answers(
    service: AgenticRagService,
    questions: list[EvaluationQuestion],
    *,
    top_k: int = 3,
) -> AnswerEvaluationReport:
    if not questions:
        raise ValueError("answer evaluation requires at least one question")

    results: list[AnswerEvaluationResult] = []
    for question in questions:
        response = service.answer(question.question, top_k=top_k)
        cited_source_ids = [
            citation.source_id for citation in response.citations
        ]
        source_hit = (
            any(
                source_id in set(question.expected_source_ids)
                for source_id in cited_source_ids
            )
            if question.expected_answerable
            else not cited_source_ids
        )
        answer_terms_matched = (
            _contains_expected_evidence(
                response.answer,
                question.expected_content_term_groups,
            )
            if question.expected_answerable
            and question.expected_content_term_groups
            else True
        )
        citations_valid = _citations_are_valid(
            response.answer,
            len(response.citations),
            required=question.expected_answerable,
        )
        passed = (
            response.status == "answered"
            and source_hit
            and answer_terms_matched
            and citations_valid
            if question.expected_answerable
            else response.status == "no_evidence" and not cited_source_ids
        )
        results.append(
            AnswerEvaluationResult(
                question_id=question.id,
                expected_answerable=question.expected_answerable,
                answer=response.answer,
                passed=passed,
                status=response.status,
                source_hit=source_hit,
                answer_terms_matched=answer_terms_matched,
                citations_valid=citations_valid,
                degraded=response.degraded,
                generation_mode=response.generation_mode,
                latency_ms=response.latency_ms,
                cited_source_ids=cited_source_ids,
            )
        )

    answerable = [
        result for result in results if result.expected_answerable
    ]
    unanswerable = [
        result for result in results if not result.expected_answerable
    ]
    latencies = sorted(result.latency_ms for result in results)
    return AnswerEvaluationReport(
        question_count=len(results),
        answerable_count=len(answerable),
        unanswerable_count=len(unanswerable),
        end_to_end_accuracy=_ratio(
            sum(result.passed for result in results),
            len(results),
        ),
        source_hit_rate=_ratio(
            sum(result.source_hit for result in answerable),
            len(answerable),
        ),
        answer_term_accuracy=_ratio(
            sum(result.answer_terms_matched for result in answerable),
            len(answerable),
        ),
        citation_validity=_ratio(
            sum(result.citations_valid for result in answerable),
            len(answerable),
        ),
        abstention_accuracy=(
            _ratio(
                sum(result.passed for result in unanswerable),
                len(unanswerable),
            )
            if unanswerable
            else None
        ),
        degraded_rate=_ratio(
            sum(result.degraded for result in results),
            len(results),
        ),
        generation_modes=dict(
            Counter(result.generation_mode for result in results)
        ),
        average_latency_ms=round(sum(latencies) / len(latencies), 2),
        p95_latency_ms=latencies[
            min(math.ceil(len(latencies) * 0.95) - 1, len(latencies) - 1)
        ],
        results=results,
    )


def _citations_are_valid(
    answer: str,
    citation_count: int,
    *,
    required: bool,
) -> bool:
    markers = [
        int(value) for value in re.findall(r"【(\d+)】", answer)
    ]
    if required and not markers:
        return False
    return all(1 <= marker <= citation_count for marker in markers)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Evaluate generated RAG answers end to end.",
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    knowledge = build_knowledge_service_from_env(args.catalog)
    try:
        report = evaluate_answers(
            AgenticRagService(
                knowledge,
                build_answer_generator_from_env(),
            ),
            load_questions(args.questions),
            top_k=args.top_k,
        )
    finally:
        knowledge.close()

    payload = report.model_dump_json(indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{payload}\n", encoding="utf-8")
    print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
