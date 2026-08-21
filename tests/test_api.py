from __future__ import annotations

from pathlib import Path
import json

from fastapi.testclient import TestClient

from firmware_knowledge_agent.api import create_app
from firmware_knowledge_agent.service import FirmwareKnowledgeService


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_client() -> TestClient:
    service = FirmwareKnowledgeService(
        PROJECT_ROOT / "data/sample/sources.json"
    )
    return TestClient(create_app(service))


def test_health_reports_loaded_corpus() -> None:
    with build_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["sources"] == 3
    assert response.json()["chunks"] >= 6
    assert response.json()["retriever"] == "bm25"
    assert response.json()["reranker"] == "none"
    assert response.json()["vector_store"] == "disabled"
    assert response.json()["generator"] == "extractive"


def test_console_and_corpus_metadata_are_available() -> None:
    with build_client() as client:
        console = client.get("/")
        favicon = client.get("/favicon.ico")
        corpus = client.get("/v1/corpus/sources")
        evaluations = client.get("/v1/evaluations")

    assert console.status_code == 200
    assert favicon.status_code == 204
    assert "Firmware Knowledge" in console.text
    assert "v=1.0.1" in console.text
    assert "PUBLIC_SAMPLE_QUESTIONS" in client.get("/static/app.js").text
    assert corpus.status_code == 200
    assert corpus.json()["sources"] == 3
    assert len(corpus.json()["items"]) == 3
    assert corpus.json()["components"]
    assert evaluations.status_code == 200
    assert any(
        report["kind"] == "retrieval"
        for report in evaluations.json()
    )


def test_search_endpoint_returns_ranked_evidence() -> None:
    with build_client() as client:
        response = client.post(
            "/v1/search",
            json={
                "query": "Wi-Fi 断开后应该如何重连？",
                "top_k": 3,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["hits"][0]["chunk"]["source_id"] == "esp-idf-wifi-events"


def test_agentic_answer_endpoint_returns_trace_and_citations() -> None:
    with build_client() as client:
        response = client.post(
            "/v1/agent/answer",
            json={
                "query": "NVS 写入后为什么要调用 nvs_commit？",
                "top_k": 3,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "answered"
    assert payload["mode"] == "agentic_rag"
    assert payload["citations"]
    assert payload["trace"] == [
        "retrieve",
        "grade_evidence",
        "generate",
        "verify_citations",
        "complete",
    ]
    assert payload["generation_mode"] == "extractive"
    assert payload["degraded"] is False
    assert payload["latency_ms"] >= 0


def test_upload_endpoint_ingests_document_and_reloads_service(
    tmp_path: Path,
) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "seed.md").write_text(
        "# Seed\n\nExisting firmware evidence for the initial corpus.",
        encoding="utf-8",
    )
    (corpus_dir / "sources.json").write_text(
        json.dumps(
            [
                {
                    "id": "seed-source",
                    "title": "Seed",
                    "path": "seed.md",
                    "source_url": "local://seed.md",
                    "version": "v1",
                    "license_note": "test",
                }
            ]
        ),
        encoding="utf-8",
    )
    service = FirmwareKnowledgeService(corpus_dir / "sources.json")

    with TestClient(create_app(service)) as client:
        response = client.post(
            "/v1/corpus/upload",
            files={
                "file": (
                    "scheduler.md",
                    (
                        "# Scheduler\n\n"
                        "Scheduled device actions are persisted and executed "
                        "by the firmware task dispatcher at the requested time."
                    ),
                    "text/markdown",
                )
            },
            data={
                "source_id": "scheduler-guide",
                "component": "scheduled-task",
            },
        )
    assert response.status_code == 200
    assert response.json()["sources"] == 2
    assert response.json()["index_rebuilt"] is True
    assert any(
        chunk.source_id == "scheduler-guide"
        for chunk in service.chunks
    )
