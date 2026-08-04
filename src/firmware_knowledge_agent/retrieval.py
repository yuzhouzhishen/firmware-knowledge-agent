from __future__ import annotations

import math
import re
from typing import Protocol

from rank_bm25 import BM25Okapi

from firmware_knowledge_agent.embeddings import (
    EmbeddingModel,
    build_chunk_texts,
)
from firmware_knowledge_agent.models import KnowledgeChunk, SearchHit


_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_+\-]+|[\u4e00-\u9fff]+")
_STOP_TOKENS = {
    "如何",
    "什么",
    "为什么",
    "应该",
    "是否",
    "可以",
    "哪个",
    "以后",
}
_MIN_QUERY_COVERAGE = 0.25
_DOMAIN_QUERY_EXPANSIONS = (
    (
        re.compile(r"固定周期|周期任务", re.IGNORECASE),
        "periodic task vTaskDelayUntil xTaskDelayUntil",
    ),
    (
        re.compile(r"周期漂移", re.IGNORECASE),
        "relative delay vTaskDelay vTaskDelayUntil",
    ),
    (
        re.compile(r"没有可用页|无可用页", re.IGNORECASE),
        "ESP_ERR_NVS_NO_FREE_PAGES",
    ),
    (
        re.compile(r"mqtt|http|tcp", re.IGNORECASE),
        "IP_EVENT_STA_GOT_IP socket TCP IP address",
    ),
    (
        re.compile(r"断线重连|断开事件|断开.*重连", re.IGNORECASE),
        (
            "WIFI_EVENT_STA_DISCONNECTED "
            "esp_wifi_connect esp_wifi_disconnect reconnect"
        ),
    ),
    (
        re.compile(r"pcName|任务名|任务名称", re.IGNORECASE),
        "pcName descriptive task name debugging aid human-readable name",
    ),
    (
        re.compile(
            r"(?:nvs.*(?:断电|掉电)|(?:断电|掉电).*nvs)",
            re.IGNORECASE,
        ),
        (
            "NVS power off loss of data new key-value pair "
            "being written"
        ),
    ),
)


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int = 3) -> list[SearchHit]: ...


class Bm25Retriever:
    def __init__(self, chunks: list[KnowledgeChunk]) -> None:
        if not chunks:
            raise ValueError("retriever requires at least one chunk")
        self._chunks = chunks
        self._corpus_tokens = [
            tokenize(f"{chunk.title} {chunk.section} {chunk.content}")
            for chunk in chunks
        ]
        self._index = BM25Okapi(self._corpus_tokens)

    def search(self, query: str, *, top_k: int = 3) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        tokens = tokenize(expand_firmware_query(query))
        if not tokens:
            return []
        unique_tokens = set(tokens)
        matched_tokens = {
            token
            for token in unique_tokens
            if any(token in document for document in self._corpus_tokens)
        }
        if len(matched_tokens) / len(unique_tokens) < _MIN_QUERY_COVERAGE:
            return []
        scores = self._index.get_scores(tokens)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        hits: list[SearchHit] = []
        for index, score in ranked[:top_k]:
            numeric_score = float(score)
            if numeric_score <= 0:
                continue
            hits.append(
                SearchHit(
                    rank=len(hits) + 1,
                    score=round(numeric_score, 6),
                    chunk=self._chunks[index],
                )
            )
        return hits


class VectorRetriever:
    def __init__(
        self,
        chunks: list[KnowledgeChunk],
        embeddings: EmbeddingModel,
        *,
        min_score: float = 0.55,
    ) -> None:
        if not chunks:
            raise ValueError("retriever requires at least one chunk")
        self._chunks = chunks
        self._embeddings = embeddings
        self._min_score = min_score
        self._vectors = embeddings.embed_documents(build_chunk_texts(chunks))

    def search(self, query: str, *, top_k: int = 3) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_vector = self._embeddings.embed_query(
            expand_firmware_query(query)
        )
        ranked = sorted(
            (
                (index, cosine_similarity(query_vector, vector))
                for index, vector in enumerate(self._vectors)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        hits: list[SearchHit] = []
        for index, score in ranked[:top_k]:
            if score < self._min_score:
                continue
            hits.append(
                SearchHit(
                    rank=len(hits) + 1,
                    score=round(score, 6),
                    chunk=self._chunks[index],
                )
            )
        return hits


class HybridRetriever:
    def __init__(
        self,
        lexical: Retriever,
        semantic: Retriever,
        *,
        rrf_k: int = 60,
    ) -> None:
        self._lexical = lexical
        self._semantic = semantic
        self._rrf_k = rrf_k

    def search(self, query: str, *, top_k: int = 3) -> list[SearchHit]:
        candidate_count = max(top_k * 3, 10)
        lexical_ranking = self._lexical.search(
            query,
            top_k=candidate_count,
        )
        semantic_ranking = self._semantic.search(
            query,
            top_k=candidate_count,
        )
        if not semantic_ranking:
            return []
        rankings = (lexical_ranking, semantic_ranking)
        fused_scores: dict[str, float] = {}
        chunks: dict[str, KnowledgeChunk] = {}
        for ranking in rankings:
            for hit in ranking:
                chunk_id = hit.chunk.id
                chunks[chunk_id] = hit.chunk
                fused_scores[chunk_id] = fused_scores.get(
                    chunk_id,
                    0.0,
                ) + 1.0 / (self._rrf_k + hit.rank)

        ordered = sorted(
            fused_scores,
            key=fused_scores.__getitem__,
            reverse=True,
        )[:top_k]
        return [
            SearchHit(
                rank=index,
                score=round(fused_scores[chunk_id], 8),
                chunk=chunks[chunk_id],
            )
            for index, chunk_id in enumerate(ordered, start=1)
        ]

    def close(self) -> None:
        for retriever in (self._lexical, self._semantic):
            close = getattr(retriever, "close", None)
            if callable(close):
                close()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding vectors must have equal non-zero dimensions")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.findall(text.lower()):
        if _contains_chinese(match):
            characters = list(match)
            if len(characters) == 1:
                tokens.append(match)
            else:
                tokens.extend(
                    "".join(characters[index : index + 2])
                    for index in range(len(characters) - 1)
                )
        else:
            tokens.append(match)
            tokens.extend(
                part
                for part in re.split(r"[_+\-]+", match)
                if part and part != match
            )
    return [token for token in tokens if token not in _STOP_TOKENS]


def expand_firmware_query(query: str) -> str:
    expansions = [
        expansion
        for pattern, expansion in _DOMAIN_QUERY_EXPANSIONS
        if pattern.search(query)
    ]
    if not expansions:
        return query
    return f"{query}\nCanonical firmware terms: {' '.join(expansions)}"


def _contains_chinese(value: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in value)
