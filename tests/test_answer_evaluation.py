from __future__ import annotations

from pathlib import Path

from firmware_knowledge_agent.agentic_workflow import (
    AgenticRagService,
    ExtractiveAnswerGenerator,
)
from firmware_knowledge_agent.answer_evaluation import (
    _citations_are_valid,
    evaluate_answers,
)
from firmware_knowledge_agent.models import EvaluationQuestion
from firmware_knowledge_agent.service import FirmwareKnowledgeService


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_answer_evaluation_covers_grounding_and_abstention() -> None:
    knowledge = FirmwareKnowledgeService(
        PROJECT_ROOT / "data/sample/sources.json"
    )
    service = AgenticRagService(
        knowledge,
        ExtractiveAnswerGenerator(),
    )
    questions = [
        EvaluationQuestion(
            id="nvs-commit",
            question="NVS 写入后为什么要调用 nvs_commit？",
            expected_source_ids=["esp-idf-nvs"],
            expected_content_term_groups=[["nvs_commit"]],
        ),
        EvaluationQuestion(
            id="out-of-domain",
            question="Zephyr devicetree overlay 怎么写？",
            expected_answerable=False,
        ),
    ]

    report = evaluate_answers(service, questions)

    assert report.question_count == 2
    assert report.end_to_end_accuracy == 1.0
    assert report.source_hit_rate == 1.0
    assert report.answer_term_accuracy == 1.0
    assert report.citation_validity == 1.0
    assert report.abstention_accuracy == 1.0
    assert report.generation_modes == {"extractive": 1, "none": 1}
    assert "nvs_commit" in report.results[0].answer


def test_citation_validator_rejects_missing_and_out_of_range_markers() -> None:
    assert not _citations_are_valid("answer", 1, required=True)
    assert not _citations_are_valid("answer 【2】", 1, required=True)
    assert _citations_are_valid("answer 【1】", 1, required=True)
