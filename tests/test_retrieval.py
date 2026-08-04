from __future__ import annotations

from pathlib import Path

from firmware_knowledge_agent.service import FirmwareKnowledgeService


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_service() -> FirmwareKnowledgeService:
    return FirmwareKnowledgeService(
        PROJECT_ROOT / "data/sample/sources.json"
    )


def test_search_returns_expected_nvs_source() -> None:
    response = build_service().search(
        "nvs_set 写入以后为什么要调用 nvs_commit",
        top_k=3,
    )

    assert response.status == "ok"
    assert response.hits[0].chunk.source_id == "esp-idf-nvs"


def test_answer_contains_citations() -> None:
    response = build_service().answer(
        "固定周期任务应该使用 vTaskDelay 还是 xTaskDelayUntil",
    )

    assert response.status == "answered"
    assert response.mode == "extractive_baseline"
    assert response.citations
    assert response.citations[0].source_id == "freertos-task-delay"


def test_unknown_question_refuses_to_answer() -> None:
    response = build_service().answer(
        "如何给 PostgreSQL 创建逻辑复制槽",
    )

    assert response.status == "no_evidence"
    assert response.citations == []
