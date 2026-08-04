from __future__ import annotations

from pathlib import Path

from firmware_knowledge_agent.agentic_workflow import (
    AgenticRagService,
    AnswerGenerationError,
)
from firmware_knowledge_agent.models import SearchHit
from firmware_knowledge_agent.service import FirmwareKnowledgeService


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RecordingGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[SearchHit]]] = []

    def generate(self, query: str, hits: list[SearchHit]) -> str:
        self.calls.append((query, hits))
        return "应等待 IP_EVENT_STA_GOT_IP 后再创建 socket。[1]"


def build_service(
    generator: RecordingGenerator,
) -> AgenticRagService:
    knowledge = FirmwareKnowledgeService(
        PROJECT_ROOT / "data/sample/sources.json"
    )
    return AgenticRagService(knowledge, generator)


def test_agentic_workflow_generates_only_after_evidence() -> None:
    generator = RecordingGenerator()
    service = build_service(generator)

    response = service.answer(
        "Wi-Fi 连接 AP 后应等待哪个事件再创建 socket？"
    )

    assert response.status == "answered"
    assert response.citations
    assert response.citations[0].source_id == "esp-idf-wifi-events"
    assert response.trace == [
        "retrieve",
        "grade_evidence",
        "generate",
        "verify_citations",
        "complete",
    ]
    assert len(generator.calls) == 1


def test_agentic_workflow_refuses_without_calling_generator() -> None:
    generator = RecordingGenerator()
    service = build_service(generator)

    response = service.answer(
        "Zephyr devicetree overlay 中 aliases 节点怎么写？"
    )

    assert response.status == "no_evidence"
    assert response.citations == []
    assert response.trace == ["retrieve", "grade_evidence", "refuse"]
    assert generator.calls == []


class MissingCitationGenerator:
    def generate(self, query: str, hits: list[SearchHit]) -> str:
        del query, hits
        return "应该等待网络拿到 IP 后再创建 socket。"


class FailingGenerator:
    def generate(self, query: str, hits: list[SearchHit]) -> str:
        del query, hits
        raise AnswerGenerationError("model is unavailable")


def test_agentic_workflow_falls_back_when_citation_is_missing() -> None:
    knowledge = FirmwareKnowledgeService(
        PROJECT_ROOT / "data/sample/sources.json"
    )
    service = AgenticRagService(
        knowledge,
        MissingCitationGenerator(),
    )

    response = service.answer(
        "Wi-Fi 连接 AP 后应等待哪个事件再创建 socket？"
    )

    assert response.status == "answered"
    assert response.answer.startswith("[1]")
    assert response.trace[-2:] == [
        "verify_citations",
        "fallback_extractive",
    ]


def test_agentic_workflow_falls_back_when_generator_fails() -> None:
    knowledge = FirmwareKnowledgeService(
        PROJECT_ROOT / "data/sample/sources.json"
    )
    service = AgenticRagService(knowledge, FailingGenerator())

    response = service.answer(
        "NVS 写入后为什么要调用 nvs_commit？"
    )

    assert response.status == "answered"
    assert response.answer.startswith("[1]")
    assert "generate_error" in response.trace
    assert response.trace[-1] == "fallback_extractive"
