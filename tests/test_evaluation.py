from __future__ import annotations

from pathlib import Path

import pytest

from firmware_knowledge_agent.evaluation import (
    audit_questions,
    evaluate,
    load_questions,
)
from firmware_knowledge_agent.models import EvaluationQuestion
from firmware_knowledge_agent.service import FirmwareKnowledgeService


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_baseline_evaluation_is_reproducible() -> None:
    service = FirmwareKnowledgeService(
        PROJECT_ROOT / "data/sample/sources.json"
    )
    questions = load_questions(PROJECT_ROOT / "data/eval/baseline.json")

    report = evaluate(service, questions, top_k=3)

    assert report.question_count == 6
    assert report.answerable_count == 6
    assert report.unanswerable_count == 0
    assert report.hit_at_k >= 0.8
    assert report.mrr >= 0.7
    assert report.abstention_accuracy is None


def test_evaluation_measures_correct_abstention() -> None:
    service = FirmwareKnowledgeService(
        PROJECT_ROOT / "data/sample/sources.json"
    )
    questions = [
        EvaluationQuestion(
            id="out-of-domain",
            question="Zephyr devicetree overlay 中 aliases 节点怎么写？",
            expected_answerable=False,
        )
    ]

    report = evaluate(service, questions, top_k=3)

    assert report.answerable_count == 0
    assert report.unanswerable_count == 1
    assert report.abstention_accuracy == 1.0
    assert report.overall_accuracy == 1.0
    assert report.results[0].abstained is True
    assert report.results[0].passed is True


def test_answerable_question_requires_expected_source() -> None:
    with pytest.raises(ValueError, match="expected_source_ids"):
        EvaluationQuestion(
            id="missing-source",
            question="NVS 如何初始化？",
        )


def test_unanswerable_question_rejects_expected_evidence() -> None:
    with pytest.raises(ValueError, match="cannot declare evidence"):
        EvaluationQuestion(
            id="invalid-unanswerable",
            question="未知问题",
            expected_answerable=False,
            expected_source_ids=["esp-idf-nvs"],
        )


def test_question_audit_accepts_reachable_evidence() -> None:
    service = FirmwareKnowledgeService(
        PROJECT_ROOT / "data/sample/sources.json"
    )
    questions = load_questions(PROJECT_ROOT / "data/eval/baseline.json")

    audit_questions(questions, service.chunks)


def test_question_audit_rejects_unreachable_evidence() -> None:
    service = FirmwareKnowledgeService(
        PROJECT_ROOT / "data/sample/sources.json"
    )
    questions = [
        EvaluationQuestion(
            id="wrong-evidence",
            question="NVS 如何初始化？",
            expected_source_ids=["esp-idf-nvs"],
            expected_content_term_groups=[["term-that-does-not-exist"]],
        )
    ]

    with pytest.raises(ValueError, match="wrong-evidence"):
        audit_questions(questions, service.chunks)


def test_question_audit_rejects_duplicate_ids() -> None:
    question = EvaluationQuestion(
        id="duplicate",
        question="未知问题",
        expected_answerable=False,
    )

    with pytest.raises(ValueError, match="duplicate"):
        audit_questions([question, question], [])
