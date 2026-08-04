from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import requests


class EmbeddingModel(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbeddingModel:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = "models/gemini-embedding-001",
        output_dimensionality: int = 768,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for vector retrieval")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self._client = GoogleGenerativeAIEmbeddings(
            model=model,
            api_key=api_key,
            output_dimensionality=output_dimensionality,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._client.embed_documents(
                texts,
                task_type="RETRIEVAL_DOCUMENT",
            )
        except Exception as exc:
            raise RuntimeError(
                "Gemini embedding failed; check API quota and billing"
            ) from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._client.embed_query(
                text,
                task_type="RETRIEVAL_QUERY",
            )
        except Exception as exc:
            raise RuntimeError(
                "Gemini embedding failed; check API quota and billing"
            ) from exc


class OllamaEmbeddingModel:
    def __init__(
        self,
        *,
        model: str = "qwen3-embedding:0.6b",
        base_url: str = "http://127.0.0.1:11434",
        batch_size: int = 16,
    ) -> None:
        self._model = model
        self._endpoint = f"{base_url.rstrip('/')}/api/embed"
        self._batch_size = batch_size
        self._query_prefix = (
            "Instruct: Given a web search query, retrieve relevant passages "
            "that answer the query\nQuery: "
            if model.startswith("qwen3-embedding")
            else ""
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), self._batch_size):
            vectors.extend(
                self._embed(texts[offset : offset + self._batch_size])
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([f"{self._query_prefix}{text}"])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = requests.post(
                self._endpoint,
                json={
                    "model": self._model,
                    "input": texts,
                },
                timeout=(3, 180),
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Ollama embedding failed; check that Ollama and the model "
                "are running locally"
            ) from exc
        payload = response.json()
        vectors = payload.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise RuntimeError("Ollama returned an invalid embedding response")
        return [
            [float(value) for value in vector]
            for vector in vectors
        ]


class CachedEmbeddingModel:
    def __init__(
        self,
        inner: EmbeddingModel,
        cache_path: Path,
        *,
        namespace: str = "gemini-embedding-001:768",
    ) -> None:
        self._inner = inner
        self._cache_path = cache_path
        self._namespace = namespace
        self._cache = self._load_cache()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(text) for text in texts]
        missing = [
            (key, text)
            for key, text in zip(keys, texts, strict=True)
            if key not in self._cache
        ]
        if missing:
            vectors = self._inner.embed_documents(
                [text for _, text in missing]
            )
            if len(vectors) != len(missing):
                raise ValueError("embedding provider returned wrong vector count")
            for (key, _), vector in zip(missing, vectors, strict=True):
                self._cache[key] = vector
            self._save_cache()
        return [self._cache[key] for key in keys]

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)

    def _key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self._namespace}:{digest}"

    def _load_cache(self) -> dict[str, list[float]]:
        if not self._cache_path.exists():
            return {}
        raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
        return {
            str(key): [float(value) for value in vector]
            for key, vector in raw.items()
        }

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._cache_path.with_suffix(
            f"{self._cache_path.suffix}.tmp"
        )
        temporary_path.write_text(
            json.dumps(self._cache, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary_path.replace(self._cache_path)


def build_chunk_texts(chunks: Sequence[object]) -> list[str]:
    texts: list[str] = []
    for chunk in chunks:
        title = getattr(chunk, "title")
        section = getattr(chunk, "section")
        content = getattr(chunk, "content")
        texts.append(f"{title}\n{section}\n{content}")
    return texts
