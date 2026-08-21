from __future__ import annotations

from pathlib import Path
from unittest import mock

from firmware_knowledge_agent.agentic_workflow import (
    AgenticRagService,
    AnswerGenerationError,
    OllamaAnswerGenerator,
)
from firmware_knowledge_agent.models import SearchHit
from firmware_knowledge_agent.service import FirmwareKnowledgeService


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RecordingGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[SearchHit]]] = []

    def generate(self, query: str, hits: list[SearchHit]) -> str:
        self.calls.append((query, hits))
        return "应等待 IP_EVENT_STA_GOT_IP 后再创建 socket。【1】"


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


class ThirdCitationGenerator:
    def generate(self, query: str, hits: list[SearchHit]) -> str:
        del query
        assert len(hits) >= 3
        return "只使用第三条检索证据。【3】"


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
    assert response.answer.startswith("【1】")
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
    assert response.answer.startswith("【1】")
    assert "generate_error" in response.trace
    assert response.trace[-1] == "fallback_extractive"


def test_agentic_workflow_returns_only_citations_used_by_answer() -> None:
    knowledge = FirmwareKnowledgeService(
        PROJECT_ROOT / "data/sample/sources.json"
    )
    expected = knowledge.search(
        "FreeRTOS 周期任务为什么适合用 vTaskDelayUntil？",
        top_k=3,
    ).hits[2].chunk.id
    service = AgenticRagService(knowledge, ThirdCitationGenerator())

    response = service.answer(
        "FreeRTOS 周期任务为什么适合用 vTaskDelayUntil？",
        top_k=3,
    )

    assert response.answer == "只使用第三条检索证据。【1】"
    assert [citation.chunk_id for citation in response.citations] == [expected]


def test_ollama_generator_renders_structured_claim_citations() -> None:
    knowledge = FirmwareKnowledgeService(
        PROJECT_ROOT / "data/sample/sources.json"
    )
    hits = knowledge.search(
        "Wi-Fi 连接 AP 后应等待哪个事件再创建 socket？",
        top_k=2,
    ).hits
    response = mock.Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "message": {
            "content": (
                '{"claims":['
                '{"text":"先等待设备取得 IP。","citations":[1]},'
                '{"text":"再创建网络 socket。","citations":[1,2]}'
                "]}"
            )
        }
    }

    with mock.patch(
        "firmware_knowledge_agent.agentic_workflow.requests.post",
        return_value=response,
    ) as post:
        answer = OllamaAnswerGenerator().generate("如何建连？", hits)

    assert answer == (
        "先等待设备取得 IP。【1】\n"
        "再创建网络 socket。【1】【2】"
    )
    request_payload = post.call_args.kwargs["json"]
    assert request_payload["format"]["properties"]["claims"]
    assert request_payload["options"]["temperature"] == 0
    assert "实时事实不是可引用文档" in (
        request_payload["messages"][0]["content"]
    )
    assert "可用文档证据编号：[1, 2]" in (
        request_payload["messages"][1]["content"]
    )
