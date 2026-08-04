from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
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


def create_app(
    service: FirmwareKnowledgeService | None = None,
    agentic_service: AgenticRagService | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owns_service = service is None
        if owns_service:
            load_dotenv()
            catalog = Path(
                os.getenv("FIRMWARE_RAG_CATALOG", str(DEFAULT_CATALOG))
            )
            retriever_mode = os.getenv(
                "FIRMWARE_RAG_RETRIEVER",
                "bm25",
            )
            if retriever_mode not in {"bm25", "vector", "hybrid"}:
                raise ValueError(
                    "FIRMWARE_RAG_RETRIEVER must be bm25, vector, or hybrid"
                )
            app.state.service = FirmwareKnowledgeService(
                catalog,
                retriever_mode=retriever_mode,
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
                reranker_mode=os.getenv(
                    "FIRMWARE_RAG_RERANKER",
                    "none",
                ),
                reranker_model=os.getenv(
                    "FIRMWARE_RAG_RERANKER_MODEL",
                    (
                        "cross-encoder/"
                        "mmarco-mMiniLMv2-L12-H384-v1"
                    ),
                ),
                reranker_model_file=os.getenv(
                    "FIRMWARE_RAG_RERANKER_ONNX_FILE",
                    "onnx/model_quint8_avx2.onnx",
                ),
                reranker_cache_dir=Path(
                    os.getenv(
                        "FIRMWARE_RAG_RERANKER_CACHE",
                        "var/models",
                    )
                ),
                reranker_candidate_count=int(
                    os.getenv(
                        "FIRMWARE_RAG_RERANKER_CANDIDATES",
                        "12",
                    )
                ),
                chunk_size=int(
                    os.getenv("FIRMWARE_RAG_CHUNK_SIZE", "700")
                ),
                chunk_overlap=int(
                    os.getenv("FIRMWARE_RAG_CHUNK_OVERLAP", "100")
                ),
                vector_min_score=float(
                    os.getenv("FIRMWARE_RAG_VECTOR_MIN_SCORE", "0.55")
                ),
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
        version="0.3.0",
        lifespan=lifespan,
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
        }

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


app = create_app()
