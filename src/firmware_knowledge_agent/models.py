from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SourceSpec(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    path: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    version: str = Field(min_length=1)
    license_note: str = Field(min_length=1)
    source_type: str = "markdown"
    component: str = "general"
    confidentiality: str = "public"


class KnowledgeChunk(BaseModel):
    id: str
    source_id: str
    title: str
    section: str
    content: str
    source_url: str
    version: str
    source_type: str = "markdown"
    component: str = "general"
    confidentiality: str = "public"


class SearchHit(BaseModel):
    rank: int
    score: float
    chunk: KnowledgeChunk


class SearchResponse(BaseModel):
    query: str
    retriever: Literal["bm25", "vector", "hybrid"]
    reranker: str = "none"
    status: Literal["ok", "no_evidence"]
    hits: list[SearchHit] = Field(default_factory=list)


class Citation(BaseModel):
    source_id: str
    title: str
    section: str
    chunk_id: str
    source_url: str


class AnswerResponse(BaseModel):
    query: str
    status: Literal["answered", "no_evidence"]
    mode: Literal["extractive_baseline"]
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class AgenticAnswerResponse(BaseModel):
    query: str
    status: Literal["answered", "no_evidence"]
    mode: Literal["agentic_rag"] = "agentic_rag"
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)


class EvaluationQuestion(BaseModel):
    id: str
    question: str
    expected_answerable: bool = True
    expected_source_ids: list[str] = Field(default_factory=list)
    expected_content_term_groups: list[list[str]] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_expected_evidence(self) -> EvaluationQuestion:
        if self.expected_answerable and not self.expected_source_ids:
            raise ValueError(
                "answerable evaluation questions require expected_source_ids"
            )
        if not self.expected_answerable and (
            self.expected_source_ids or self.expected_content_term_groups
        ):
            raise ValueError(
                "unanswerable evaluation questions cannot declare evidence"
            )
        return self


class EvaluationResult(BaseModel):
    question_id: str
    expected_answerable: bool
    passed: bool
    hit: bool
    abstained: bool
    reciprocal_rank: float
    retrieved_source_ids: list[str]
    retrieved_chunk_ids: list[str]


class EvaluationRunConfig(BaseModel):
    retriever: Literal["bm25", "vector", "hybrid"]
    reranker: str
    embedding_provider: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    vector_min_score: float
    strict_label_audit: bool
    catalog_sha256: str
    questions_sha256: str


class EvaluationReport(BaseModel):
    top_k: int
    question_count: int
    answerable_count: int
    unanswerable_count: int
    hit_at_k: float
    mrr: float
    abstention_accuracy: float | None
    overall_accuracy: float
    results: list[EvaluationResult]
    run_config: EvaluationRunConfig | None = None
