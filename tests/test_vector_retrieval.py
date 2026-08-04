from __future__ import annotations

from pathlib import Path

from firmware_knowledge_agent.service import FirmwareKnowledgeService


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TopicEmbeddingModel:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        normalized = text.lower()
        if (
            "nvs" in normalized
            or "配置存储" in normalized
            or "设备配置" in normalized
        ):
            return [1.0, 0.0, 0.0]
        if "wi-fi" in normalized or "重连" in normalized:
            return [0.0, 1.0, 0.0]
        if "freertos" in normalized or "周期任务" in normalized:
            return [0.0, 0.0, 1.0]
        return [0.0, 0.0, 0.0]


def test_vector_retriever_can_use_semantic_provider() -> None:
    service = FirmwareKnowledgeService(
        PROJECT_ROOT / "data/sample/sources.json",
        retriever_mode="vector",
        embeddings=TopicEmbeddingModel(),
    )

    response = service.search("如何保存设备配置", top_k=3)

    assert response.retriever == "vector"
    assert response.hits[0].chunk.source_id == "esp-idf-nvs"


def test_hybrid_retriever_fuses_lexical_and_vector_results() -> None:
    service = FirmwareKnowledgeService(
        PROJECT_ROOT / "data/sample/sources.json",
        retriever_mode="hybrid",
        embeddings=TopicEmbeddingModel(),
    )

    response = service.search("Wi-Fi 断开后如何重连", top_k=3)

    assert response.retriever == "hybrid"
    assert response.hits[0].chunk.source_id == "esp-idf-wifi-events"


def test_vector_retriever_rejects_low_similarity_query() -> None:
    service = FirmwareKnowledgeService(
        PROJECT_ROOT / "data/sample/sources.json",
        retriever_mode="vector",
        embeddings=TopicEmbeddingModel(),
    )

    response = service.search("Zephyr devicetree overlay", top_k=3)

    assert response.status == "no_evidence"
    assert response.hits == []


def test_hybrid_retriever_requires_semantic_evidence() -> None:
    service = FirmwareKnowledgeService(
        PROJECT_ROOT / "data/sample/sources.json",
        retriever_mode="hybrid",
        embeddings=TopicEmbeddingModel(),
    )

    response = service.search(
        "LVGL display callback 如何注册",
        top_k=3,
    )

    assert response.status == "no_evidence"
    assert response.hits == []
