from __future__ import annotations

from firmware_knowledge_agent.models import KnowledgeChunk, SearchHit
from firmware_knowledge_agent.reranking import RerankingRetriever


class FixedRetriever:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.requested_top_k = 0

    def search(self, query: str, *, top_k: int = 3) -> list[SearchHit]:
        del query
        self.requested_top_k = top_k
        return self.hits[:top_k]


class TermReranker:
    def rerank(
        self,
        query: str,
        hits: list[SearchHit],
        *,
        top_k: int,
    ) -> list[SearchHit]:
        term = query.casefold()
        ordered = sorted(
            hits,
            key=lambda hit: term in hit.chunk.content.casefold(),
            reverse=True,
        )[:top_k]
        return [
            SearchHit(rank=index, score=1.0, chunk=hit.chunk)
            for index, hit in enumerate(ordered, start=1)
        ]


def _hit(source_id: str, content: str, rank: int) -> SearchHit:
    return SearchHit(
        rank=rank,
        score=1.0 / rank,
        chunk=KnowledgeChunk(
            id=f"{source_id}:0001",
            source_id=source_id,
            title=source_id,
            section="section",
            content=content,
            source_url=f"local://{source_id}",
            version="v1",
        ),
    )


def test_reranking_reorders_candidates_and_limits_results() -> None:
    inner = FixedRetriever(
        [
            _hit("generic", "generic device notes", 1),
            _hit("mqtt", "mqtt reconnect strategy", 2),
            _hit("timer", "scheduled task notes", 3),
        ]
    )
    retriever = RerankingRetriever(
        inner,
        TermReranker(),
        candidate_count=12,
    )

    hits = retriever.search("mqtt", top_k=2)

    assert inner.requested_top_k == 12
    assert [hit.chunk.source_id for hit in hits] == ["mqtt", "generic"]
    assert [hit.rank for hit in hits] == [1, 2]
