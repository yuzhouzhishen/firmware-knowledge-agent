from __future__ import annotations

from pathlib import Path

from firmware_knowledge_agent.models import KnowledgeChunk
from firmware_knowledge_agent.vector_store import QdrantVectorRetriever


class CountingEmbeddingModel:
    def __init__(self) -> None:
        self.document_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        if "MQTT" in text or "重连" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]


def _chunks() -> list[KnowledgeChunk]:
    return [
        KnowledgeChunk(
            id="mqtt:0001",
            source_id="mqtt",
            title="MQTT 连接",
            section="重连",
            content="连接断开后执行退避重连。",
            source_url="https://example.com/mqtt",
            version="v1",
            component="connectivity",
        ),
        KnowledgeChunk(
            id="timer:0001",
            source_id="timer",
            title="定时任务",
            section="调度",
            content="任务由调度器按计划触发。",
            source_url="https://example.com/timer",
            version="v1",
            component="scheduler",
        ),
    ]


def test_qdrant_index_persists_and_is_reused(tmp_path: Path) -> None:
    first_embeddings = CountingEmbeddingModel()
    first = QdrantVectorRetriever(
        _chunks(),
        first_embeddings,
        storage_path=tmp_path / "qdrant",
        embedding_namespace="test:topics",
        min_score=0.5,
    )

    hits = first.search("MQTT 断线后如何重连")

    assert hits[0].chunk.source_id == "mqtt"
    assert hits[0].chunk.component == "connectivity"
    assert first_embeddings.document_calls == 1
    first.close()

    second_embeddings = CountingEmbeddingModel()
    second = QdrantVectorRetriever(
        _chunks(),
        second_embeddings,
        storage_path=tmp_path / "qdrant",
        embedding_namespace="test:topics",
        min_score=0.5,
    )

    second.search("定时任务如何调度")

    assert second_embeddings.document_calls == 0
    second.close()
