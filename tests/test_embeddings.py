from __future__ import annotations

from typing import Any

from firmware_knowledge_agent.embeddings import OllamaEmbeddingModel


class FakeResponse:
    def __init__(
        self,
        vectors: list[list[float]] | None = None,
    ) -> None:
        self._vectors = vectors or [[1.0, 0.0], [0.0, 1.0]]

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, list[list[float]]]:
        return {"embeddings": self._vectors}


def test_ollama_embedding_provider_uses_batch_endpoint(
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(
        url: str,
        *,
        json: dict[str, Any],
        timeout: tuple[int, int],
    ) -> FakeResponse:
        calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr(
        "firmware_knowledge_agent.embeddings.requests.post",
        fake_post,
    )
    model = OllamaEmbeddingModel(model="test-embedding")

    vectors = model.embed_documents(["first", "second"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert calls[0]["json"] == {
        "model": "test-embedding",
        "input": ["first", "second"],
    }


def test_qwen_query_uses_retrieval_instruction(
    monkeypatch: Any,
) -> None:
    payloads: list[dict[str, Any]] = []

    def fake_post(
        url: str,
        *,
        json: dict[str, Any],
        timeout: tuple[int, int],
    ) -> FakeResponse:
        payloads.append(json)
        return FakeResponse([[1.0, 0.0]])

    monkeypatch.setattr(
        "firmware_knowledge_agent.embeddings.requests.post",
        fake_post,
    )
    model = OllamaEmbeddingModel(model="qwen3-embedding:0.6b")

    model.embed_query("何时启动 MQTT？")

    assert payloads[0]["input"][0].startswith("Instruct:")
    assert payloads[0]["input"][0].endswith("何时启动 MQTT？")
