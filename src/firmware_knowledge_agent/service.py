from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, cast

from firmware_knowledge_agent.corpus import load_corpus
from firmware_knowledge_agent.embeddings import (
    CachedEmbeddingModel,
    EmbeddingModel,
    GeminiEmbeddingModel,
    OllamaEmbeddingModel,
)
from firmware_knowledge_agent.models import (
    AnswerResponse,
    Citation,
    KnowledgeChunk,
    SearchResponse,
)
from firmware_knowledge_agent.retrieval import (
    Bm25Retriever,
    HybridRetriever,
    Retriever,
    VectorRetriever,
)
from firmware_knowledge_agent.reranking import (
    CrossEncoderReranker,
    Reranker,
    RerankingRetriever,
)
from firmware_knowledge_agent.vector_store import QdrantVectorRetriever


class FirmwareKnowledgeService:
    def __init__(
        self,
        catalog_path: Path,
        *,
        retriever_mode: Literal["bm25", "vector", "hybrid"] = "bm25",
        embeddings: EmbeddingModel | None = None,
        embedding_cache_path: Path | None = None,
        vector_store_path: Path | None = None,
        vector_collection: str = "firmware_knowledge",
        reranker_mode: Literal["none", "cross-encoder"] = "none",
        reranker: Reranker | None = None,
        reranker_model: str = (
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        ),
        reranker_model_file: str = "onnx/model_quint8_avx2.onnx",
        reranker_cache_dir: Path = Path("var/models"),
        reranker_candidate_count: int = 12,
        chunk_size: int = 700,
        chunk_overlap: int = 100,
        vector_min_score: float = 0.55,
    ) -> None:
        if not 0.0 <= vector_min_score <= 1.0:
            raise ValueError("vector_min_score must be between 0 and 1")
        if reranker_mode not in {"none", "cross-encoder"}:
            raise ValueError(
                "reranker_mode must be none or cross-encoder"
            )
        if reranker_candidate_count <= 0:
            raise ValueError(
                "reranker_candidate_count must be positive"
            )
        self.catalog_path = catalog_path.resolve()
        self.retriever_mode = retriever_mode
        self.reranker_mode = reranker_mode
        self.vector_min_score = vector_min_score
        self._configured_embeddings = embeddings
        self._embedding_cache_path = embedding_cache_path
        self._vector_store_path = vector_store_path
        self._vector_collection = vector_collection
        self._configured_reranker = reranker
        self._reranker_model = reranker_model
        self._reranker_model_file = reranker_model_file
        self._reranker_cache_dir = reranker_cache_dir
        self._reranker_candidate_count = reranker_candidate_count
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self.chunks = []
        self.source_count = 0
        self.vector_store_backend = (
            "disabled" if retriever_mode == "bm25" else "memory"
        )
        self._retriever: Retriever
        self.reload()

    def reload(self) -> None:
        previous_retriever = getattr(self, "_retriever", None)
        if previous_retriever is not None:
            close = getattr(previous_retriever, "close", None)
            if callable(close):
                close()
        self.chunks = load_corpus(
            self.catalog_path,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        self.source_count = len({chunk.source_id for chunk in self.chunks})
        self.vector_store_backend = (
            "disabled" if self.retriever_mode == "bm25" else "memory"
        )
        self._retriever = self._build_retriever(
            self._configured_embeddings,
            embedding_cache_path=self._embedding_cache_path,
            vector_store_path=self._vector_store_path,
            vector_collection=self._vector_collection,
        )

    def close(self) -> None:
        close = getattr(self._retriever, "close", None)
        if callable(close):
            close()

    def source_summaries(self) -> list[dict[str, str]]:
        by_source: dict[str, KnowledgeChunk] = {}
        for chunk in self.chunks:
            by_source.setdefault(chunk.source_id, chunk)
        return [
            {
                "source_id": chunk.source_id,
                "title": chunk.title,
                "source_url": chunk.source_url,
                "version": chunk.version,
                "source_type": chunk.source_type,
                "component": chunk.component,
                "confidentiality": chunk.confidentiality,
            }
            for chunk in sorted(
                by_source.values(),
                key=lambda item: (item.component, item.title),
            )
        ]

    def search(self, query: str, *, top_k: int = 3) -> SearchResponse:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        hits = self._retriever.search(normalized_query, top_k=top_k)
        return SearchResponse(
            query=normalized_query,
            retriever=self.retriever_mode,
            reranker=self.reranker_mode,
            status="ok" if hits else "no_evidence",
            hits=hits,
        )

    def answer(self, query: str, *, top_k: int = 3) -> AnswerResponse:
        search = self.search(query, top_k=top_k)
        if not search.hits:
            return AnswerResponse(
                query=search.query,
                status="no_evidence",
                mode="extractive_baseline",
                answer="当前知识库没有找到足够证据，暂不回答。",
            )

        selected = search.hits[:2]
        answer = "\n\n".join(
            f"[{index}] {hit.chunk.content}"
            for index, hit in enumerate(selected, start=1)
        )
        citations = [
            Citation(
                source_id=hit.chunk.source_id,
                title=hit.chunk.title,
                section=hit.chunk.section,
                chunk_id=hit.chunk.id,
                source_url=hit.chunk.source_url,
            )
            for hit in selected
        ]
        return AnswerResponse(
            query=search.query,
            status="answered",
            mode="extractive_baseline",
            answer=answer,
            citations=citations,
        )

    def _build_retriever(
        self,
        embeddings: EmbeddingModel | None,
        *,
        embedding_cache_path: Path | None,
        vector_store_path: Path | None,
        vector_collection: str,
    ) -> Retriever:
        lexical = Bm25Retriever(self.chunks)
        if self.retriever_mode == "bm25":
            retriever: Retriever = lexical
        else:
            provider = os.getenv(
                "FIRMWARE_RAG_EMBEDDING_PROVIDER",
                "gemini",
            ).lower()
            model_name = os.getenv(
                "FIRMWARE_RAG_EMBEDDING_MODEL",
                (
                    "qwen3-embedding:0.6b"
                    if provider == "ollama"
                    else "models/gemini-embedding-001"
                ),
            )
            if embeddings is None:
                if provider == "gemini":
                    embeddings = GeminiEmbeddingModel(
                        api_key=os.getenv("GEMINI_API_KEY", ""),
                        model=model_name,
                    )
                elif provider == "ollama":
                    embeddings = OllamaEmbeddingModel(
                        model=model_name,
                        base_url=os.getenv(
                            "FIRMWARE_RAG_OLLAMA_URL",
                            "http://127.0.0.1:11434",
                        ),
                    )
                else:
                    raise ValueError(
                        "FIRMWARE_RAG_EMBEDDING_PROVIDER must be "
                        "gemini or ollama"
                    )
            if embedding_cache_path is not None:
                embeddings = CachedEmbeddingModel(
                    embeddings,
                    embedding_cache_path,
                    namespace=f"{provider}:{model_name}",
                )
            if vector_store_path is None:
                semantic: Retriever = VectorRetriever(
                    self.chunks,
                    embeddings,
                    min_score=self.vector_min_score,
                )
            else:
                semantic = QdrantVectorRetriever(
                    self.chunks,
                    embeddings,
                    storage_path=vector_store_path,
                    collection_name=vector_collection,
                    embedding_namespace=f"{provider}:{model_name}",
                    min_score=self.vector_min_score,
                )
                self.vector_store_backend = "qdrant-local"
            if self.retriever_mode == "vector":
                retriever = semantic
            elif self.retriever_mode == "hybrid":
                retriever = HybridRetriever(lexical, semantic)
            else:
                raise ValueError(
                    f"unsupported retriever mode: {self.retriever_mode}"
                )
        if self.reranker_mode == "none":
            return retriever
        if self._configured_reranker is None:
            self._configured_reranker = CrossEncoderReranker(
                self._reranker_model,
                model_file=self._reranker_model_file,
                cache_dir=self._reranker_cache_dir,
            )
        return RerankingRetriever(
            retriever,
            self._configured_reranker,
            candidate_count=self._reranker_candidate_count,
        )


def build_knowledge_service_from_env(
    catalog_path: Path,
) -> FirmwareKnowledgeService:
    retriever = os.getenv("FIRMWARE_RAG_RETRIEVER", "bm25").lower()
    if retriever not in {"bm25", "vector", "hybrid"}:
        raise ValueError(
            "FIRMWARE_RAG_RETRIEVER must be bm25, vector, or hybrid"
        )
    reranker = os.getenv("FIRMWARE_RAG_RERANKER", "none").lower()
    if reranker not in {"none", "cross-encoder"}:
        raise ValueError(
            "FIRMWARE_RAG_RERANKER must be none or cross-encoder"
        )
    return FirmwareKnowledgeService(
        catalog_path,
        retriever_mode=cast(
            Literal["bm25", "vector", "hybrid"],
            retriever,
        ),
        embedding_cache_path=Path(
            os.getenv(
                "FIRMWARE_RAG_EMBEDDING_CACHE",
                "var/embeddings.json",
            )
        ),
        vector_store_path=Path(
            os.getenv(
                "FIRMWARE_RAG_VECTOR_STORE",
                "var/vector-store",
            )
        ),
        vector_collection=os.getenv(
            "FIRMWARE_RAG_VECTOR_COLLECTION",
            "firmware_knowledge",
        ),
        reranker_mode=cast(
            Literal["none", "cross-encoder"],
            reranker,
        ),
        reranker_model=os.getenv(
            "FIRMWARE_RAG_RERANKER_MODEL",
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        ),
        reranker_model_file=os.getenv(
            "FIRMWARE_RAG_RERANKER_ONNX_FILE",
            "onnx/model_quint8_avx2.onnx",
        ),
        reranker_cache_dir=Path(
            os.getenv("FIRMWARE_RAG_RERANKER_CACHE", "var/models")
        ),
        reranker_candidate_count=int(
            os.getenv("FIRMWARE_RAG_RERANKER_CANDIDATES", "12")
        ),
        chunk_size=int(os.getenv("FIRMWARE_RAG_CHUNK_SIZE", "700")),
        chunk_overlap=int(
            os.getenv("FIRMWARE_RAG_CHUNK_OVERLAP", "100")
        ),
        vector_min_score=float(
            os.getenv("FIRMWARE_RAG_VECTOR_MIN_SCORE", "0.55")
        ),
    )
