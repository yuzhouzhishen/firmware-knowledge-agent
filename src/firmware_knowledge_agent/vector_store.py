from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

from firmware_knowledge_agent.embeddings import (
    EmbeddingModel,
    build_chunk_texts,
)
from firmware_knowledge_agent.models import KnowledgeChunk, SearchHit
from firmware_knowledge_agent.retrieval import expand_firmware_query


_INDEX_SCHEMA_VERSION = 1


class QdrantVectorRetriever:
    def __init__(
        self,
        chunks: list[KnowledgeChunk],
        embeddings: EmbeddingModel,
        *,
        storage_path: Path,
        collection_name: str = "firmware_knowledge",
        embedding_namespace: str,
        min_score: float = 0.55,
    ) -> None:
        if not chunks:
            raise ValueError("retriever requires at least one chunk")
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")

        self._embeddings = embeddings
        self._min_score = min_score
        self._collection_name = collection_name
        self._storage_path = storage_path.resolve()
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(
            path=str(self._storage_path),
            force_disable_check_same_thread=True,
        )
        self._ensure_index(
            chunks,
            embedding_namespace=embedding_namespace,
        )

    def search(self, query: str, *, top_k: int = 3) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_vector = self._embeddings.embed_query(
            expand_firmware_query(query)
        )
        points = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=self._min_score,
            with_payload=True,
        ).points
        return [
            SearchHit(
                rank=rank,
                score=round(float(point.score), 6),
                chunk=KnowledgeChunk.model_validate(point.payload or {}),
            )
            for rank, point in enumerate(points, start=1)
        ]

    def close(self) -> None:
        self._client.close()

    def _ensure_index(
        self,
        chunks: list[KnowledgeChunk],
        *,
        embedding_namespace: str,
    ) -> None:
        fingerprint = _index_fingerprint(chunks, embedding_namespace)
        manifest_path = self._manifest_path()
        if (
            self._client.collection_exists(self._collection_name)
            and _read_manifest(manifest_path).get("fingerprint")
            == fingerprint
        ):
            return

        if self._client.collection_exists(self._collection_name):
            self._client.delete_collection(self._collection_name)

        vectors = self._embeddings.embed_documents(build_chunk_texts(chunks))
        if len(vectors) != len(chunks):
            raise ValueError("embedding provider returned wrong vector count")
        if not vectors or not vectors[0]:
            raise ValueError("embedding provider returned empty vectors")
        vector_size = len(vectors[0])
        if any(len(vector) != vector_size for vector in vectors):
            raise ValueError("embedding vectors must use one dimension")

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        self._client.upsert(
            collection_name=self._collection_name,
            wait=True,
            points=[
                models.PointStruct(
                    id=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"firmware-rag:{chunk.id}",
                        )
                    ),
                    vector=vector,
                    payload=chunk.model_dump(mode="json"),
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )
        _write_manifest(
            manifest_path,
            {
                "schema_version": _INDEX_SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "embedding_namespace": embedding_namespace,
                "chunk_count": len(chunks),
                "vector_size": vector_size,
            },
        )

    def _manifest_path(self) -> Path:
        return self._storage_path / (
            f".{self._collection_name}.manifest.json"
        )


def _index_fingerprint(
    chunks: list[KnowledgeChunk],
    embedding_namespace: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"schema:{_INDEX_SCHEMA_VERSION}\n".encode())
    digest.update(f"embedding:{embedding_namespace}\n".encode())
    for chunk in chunks:
        digest.update(chunk.model_dump_json().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
