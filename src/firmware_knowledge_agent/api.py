from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, AsyncIterator

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from firmware_knowledge_agent.agentic_workflow import (
    AgenticRagService,
    build_answer_generator_from_env,
)
from firmware_knowledge_agent.evaluation import DEFAULT_CATALOG
from firmware_knowledge_agent.local_ingestion import (
    MAX_DOCUMENT_BYTES,
    ingest_document_bytes,
)
from firmware_knowledge_agent.models import (
    AgenticAnswerResponse,
    AnswerResponse,
    SearchResponse,
)
from firmware_knowledge_agent.service import FirmwareKnowledgeService
from firmware_knowledge_agent.service import (
    build_knowledge_service_from_env,
)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)


class CorpusUploadResponse(BaseModel):
    source_id: str
    title: str
    created: bool
    sources: int
    chunks: int
    index_rebuilt: bool


class SourceSummary(BaseModel):
    source_id: str
    title: str
    source_url: str
    version: str
    source_type: str
    component: str
    confidentiality: str


class CorpusSummary(BaseModel):
    sources: int
    chunks: int
    components: dict[str, int]
    items: list[SourceSummary]


class EvaluationSummary(BaseModel):
    name: str
    kind: str
    question_count: int
    metrics: dict[str, float | None]


def create_app(
    service: FirmwareKnowledgeService | None = None,
    agentic_service: AgenticRagService | None = None,
) -> FastAPI:
    web_directory = Path(__file__).with_name("web")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owns_service = service is None
        if owns_service:
            load_dotenv()
            catalog = Path(
                os.getenv("FIRMWARE_RAG_CATALOG", str(DEFAULT_CATALOG))
            )
            app.state.service = build_knowledge_service_from_env(
                catalog
            )
        else:
            app.state.service = service
        app.state.agentic_service = (
            agentic_service
            or AgenticRagService(
                app.state.service,
                build_answer_generator_from_env(),
            )
        )
        try:
            yield
        finally:
            if owns_service:
                app.state.service.close()

    app = FastAPI(
        title="Firmware Knowledge Agent API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.mount(
        "/static",
        StaticFiles(directory=web_directory),
        name="static",
    )
    if service is not None:
        app.state.service = service
        app.state.agentic_service = (
            agentic_service
            or AgenticRagService(
                service,
                build_answer_generator_from_env(),
            )
        )

    @app.get("/", include_in_schema=False)
    async def console() -> FileResponse:
        return FileResponse(web_directory / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/health")
    async def health(
        request: Request,
    ) -> dict[str, int | str | float]:
        current = _service(request)
        return {
            "status": "ok",
            "sources": current.source_count,
            "chunks": len(current.chunks),
            "retriever": current.retriever_mode,
            "reranker": current.reranker_mode,
            "vector_store": current.vector_store_backend,
            "vector_min_score": current.vector_min_score,
            "agentic_workflow": "ready",
            "generator": os.getenv(
                "FIRMWARE_RAG_GENERATOR",
                "extractive",
            ),
        }

    @app.get(
        "/v1/corpus/sources",
        response_model=CorpusSummary,
    )
    async def corpus_sources(request: Request) -> CorpusSummary:
        current = _service(request)
        items = [
            SourceSummary.model_validate(item)
            for item in current.source_summaries()
        ]
        components: dict[str, int] = {}
        for item in items:
            components[item.component] = (
                components.get(item.component, 0) + 1
            )
        return CorpusSummary(
            sources=current.source_count,
            chunks=len(current.chunks),
            components=dict(sorted(components.items())),
            items=items,
        )

    @app.get(
        "/v1/evaluations",
        response_model=list[EvaluationSummary],
    )
    async def evaluation_summaries() -> list[EvaluationSummary]:
        return _load_evaluation_summaries()

    @app.post("/v1/search", response_model=SearchResponse)
    async def search(
        body: QueryRequest,
        request: Request,
    ) -> SearchResponse:
        try:
            return _service(request).search(
                body.query,
                top_k=body.top_k,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/answer", response_model=AnswerResponse)
    async def answer(
        body: QueryRequest,
        request: Request,
    ) -> AnswerResponse:
        try:
            return _service(request).answer(
                body.query,
                top_k=body.top_k,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(
        "/v1/agent/answer",
        response_model=AgenticAnswerResponse,
    )
    async def agentic_answer(
        body: QueryRequest,
        request: Request,
    ) -> AgenticAnswerResponse:
        try:
            return _agentic_service(request).answer(
                body.query,
                top_k=body.top_k,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(
        "/v1/corpus/upload",
        response_model=CorpusUploadResponse,
    )
    async def upload_corpus_document(
        request: Request,
        file: Annotated[UploadFile, File(description="Private document")],
        source_id: Annotated[str | None, Form()] = None,
        title: Annotated[str | None, Form()] = None,
        source_url: Annotated[str | None, Form()] = None,
        component: Annotated[str, Form()] = "general",
        confidentiality: Annotated[str, Form()] = "private-local",
    ) -> CorpusUploadResponse:
        content = await file.read(MAX_DOCUMENT_BYTES + 1)
        current = _service(request)
        try:
            result = ingest_document_bytes(
                filename=file.filename or "",
                content=content,
                catalog_path=current.catalog_path,
                source_id=source_id,
                title=title,
                source_url=source_url,
                component=component,
                confidentiality=confidentiality,
            )
            current.reload()
        except (UnicodeDecodeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return CorpusUploadResponse(
            source_id=result.source_id,
            title=result.title,
            created=result.created,
            sources=current.source_count,
            chunks=len(current.chunks),
            index_rebuilt=True,
        )

    return app


def _service(request: Request) -> FirmwareKnowledgeService:
    try:
        return request.app.state.service
    except AttributeError as exc:
        raise HTTPException(
            status_code=503,
            detail="service is not initialized",
        ) from exc


def _agentic_service(request: Request) -> AgenticRagService:
    try:
        return request.app.state.agentic_service
    except AttributeError as exc:
        raise HTTPException(
            status_code=503,
            detail="agentic service is not initialized",
        ) from exc


def _load_evaluation_summaries() -> list[EvaluationSummary]:
    project_root = Path(__file__).resolve().parents[2]
    configured = os.getenv("FIRMWARE_RAG_EVAL_REPORTS", "")
    paths = (
        [Path(value.strip()) for value in configured.split(",") if value.strip()]
        if configured
        else [
            project_root / "evals/reports/retrieval_v1_bm25.json",
            project_root / "evals/reports/retrieval_v1_vector.json",
            project_root
            / "evals/reports/retrieval_v1_vector_query_expansion.json",
            project_root / "evals/reports/holdout_v1_vector.json",
            project_root / "var/private-corpus/holdout-v2-vector.json",
            project_root
            / "var/private-corpus/answer-eval-extractive.json",
            project_root / "var/private-corpus/answer-eval-ollama.json",
        ]
    )
    summaries: list[EvaluationSummary] = []
    for path in paths:
        if not path.is_absolute():
            path = project_root / path
        if not path.exists():
            continue
        try:
            payload: dict[str, Any] = json.loads(
                path.read_text(encoding="utf-8")
            )
        except (OSError, TypeError, ValueError):
            continue
        if "end_to_end_accuracy" in payload:
            metrics = {
                key: payload.get(key)
                for key in (
                    "end_to_end_accuracy",
                    "source_hit_rate",
                    "answer_term_accuracy",
                    "citation_validity",
                    "abstention_accuracy",
                    "degraded_rate",
                    "average_latency_ms",
                    "p95_latency_ms",
                )
            }
            kind = "answer"
        else:
            metrics = {
                key: payload.get(key)
                for key in (
                    "hit_at_k",
                    "mrr",
                    "abstention_accuracy",
                    "overall_accuracy",
                )
            }
            kind = "retrieval"
        summaries.append(
            EvaluationSummary(
                name=path.stem,
                kind=kind,
                question_count=int(payload.get("question_count", 0)),
                metrics=metrics,
            )
        )
    return summaries


app = create_app()
