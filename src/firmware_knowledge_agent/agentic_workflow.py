from __future__ import annotations

import operator
import os
import re
from typing import Annotated, Literal, Protocol, TypedDict

import requests
from langgraph.graph import END, START, StateGraph

from firmware_knowledge_agent.models import (
    AgenticAnswerResponse,
    Citation,
    SearchHit,
    SearchResponse,
)
from firmware_knowledge_agent.service import FirmwareKnowledgeService


class AnswerGenerator(Protocol):
    def generate(self, query: str, hits: list[SearchHit]) -> str: ...


class AnswerGenerationError(RuntimeError):
    pass


class ExtractiveAnswerGenerator:
    def generate(self, query: str, hits: list[SearchHit]) -> str:
        del query
        return "\n\n".join(
            f"[{index}] {hit.chunk.content}"
            for index, hit in enumerate(hits[:2], start=1)
        )


class OllamaAnswerGenerator:
    def __init__(
        self,
        *,
        model: str = "llama3.1:8b",
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 90.0,
    ) -> None:
        self._model = model
        self._endpoint = f"{base_url.rstrip('/')}/api/chat"
        self._timeout_seconds = timeout_seconds

    def generate(self, query: str, hits: list[SearchHit]) -> str:
        evidence = "\n\n".join(
            (
                f"[{index}] source={hit.chunk.source_id}; "
                f"section={hit.chunk.section}\n{hit.chunk.content}"
            )
            for index, hit in enumerate(hits, start=1)
        )
        try:
            response = requests.post(
                self._endpoint,
                json={
                    "model": self._model,
                    "stream": False,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "你是固件文档助手。只能依据给定证据回答；"
                                "不要补充证据中没有的事实。关键结论后使用 "
                                "[1]、[2] 形式标注证据编号。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"问题：{query}\n\n"
                                f"证据：\n{evidence}"
                            ),
                        },
                    ],
                },
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            answer = str(
                payload.get("message", {}).get("content", "")
            ).strip()
        except (requests.RequestException, TypeError, ValueError) as exc:
            raise AnswerGenerationError(str(exc)) from exc
        if not answer:
            raise AnswerGenerationError("Ollama returned an empty answer")
        return answer


class AgenticRagState(TypedDict, total=False):
    query: str
    top_k: int
    search: SearchResponse
    has_evidence: bool
    answer: str
    answer_valid: bool
    citations: list[Citation]
    status: Literal["answered", "no_evidence"]
    trace: Annotated[list[str], operator.add]


class AgenticRagService:
    def __init__(
        self,
        knowledge: FirmwareKnowledgeService,
        generator: AnswerGenerator,
    ) -> None:
        self._knowledge = knowledge
        self._generator = generator
        self._graph = self._build_graph()

    def answer(
        self,
        query: str,
        *,
        top_k: int = 3,
    ) -> AgenticAnswerResponse:
        state = self._graph.invoke(
            {
                "query": query,
                "top_k": top_k,
                "trace": [],
            }
        )
        return AgenticAnswerResponse(
            query=query.strip(),
            status=state["status"],
            answer=state["answer"],
            citations=state.get("citations", []),
            trace=state.get("trace", []),
        )

    def _build_graph(self):
        def retrieve(state: AgenticRagState) -> AgenticRagState:
            return {
                "search": self._knowledge.search(
                    state["query"],
                    top_k=state["top_k"],
                ),
                "trace": ["retrieve"],
            }

        def grade_evidence(state: AgenticRagState) -> AgenticRagState:
            search = state["search"]
            citations = [
                Citation(
                    source_id=hit.chunk.source_id,
                    title=hit.chunk.title,
                    section=hit.chunk.section,
                    chunk_id=hit.chunk.id,
                    source_url=hit.chunk.source_url,
                )
                for hit in search.hits
            ]
            return {
                "has_evidence": search.status == "ok" and bool(search.hits),
                "citations": citations,
                "trace": ["grade_evidence"],
            }

        def route_evidence(
            state: AgenticRagState,
        ) -> Literal["generate", "refuse"]:
            return "generate" if state["has_evidence"] else "refuse"

        def generate(state: AgenticRagState) -> AgenticRagState:
            try:
                answer = self._generator.generate(
                    state["query"],
                    state["search"].hits,
                )
                trace = ["generate"]
            except AnswerGenerationError:
                answer = ""
                trace = ["generate_error"]
            return {
                "status": "answered",
                "answer": answer,
                "trace": trace,
            }

        def verify_answer(state: AgenticRagState) -> AgenticRagState:
            markers = [
                int(value)
                for value in re.findall(r"\[(\d+)\]", state["answer"])
            ]
            answer_valid = bool(markers) and all(
                1 <= marker <= len(state["search"].hits)
                for marker in markers
            )
            return {
                "answer_valid": answer_valid,
                "trace": ["verify_citations"],
            }

        def route_answer(
            state: AgenticRagState,
        ) -> Literal["complete", "fallback"]:
            return "complete" if state["answer_valid"] else "fallback"

        def complete(state: AgenticRagState) -> AgenticRagState:
            del state
            return {"trace": ["complete"]}

        def fallback(state: AgenticRagState) -> AgenticRagState:
            return {
                "status": "answered",
                "answer": ExtractiveAnswerGenerator().generate(
                    state["query"],
                    state["search"].hits,
                ),
                "trace": ["fallback_extractive"],
            }

        def refuse(state: AgenticRagState) -> AgenticRagState:
            del state
            return {
                "status": "no_evidence",
                "answer": "当前知识库没有找到足够证据，暂不回答。",
                "citations": [],
                "trace": ["refuse"],
            }

        builder = StateGraph(AgenticRagState)
        builder.add_node("retrieve", retrieve)
        builder.add_node("grade_evidence", grade_evidence)
        builder.add_node("generate", generate)
        builder.add_node("verify_answer", verify_answer)
        builder.add_node("complete", complete)
        builder.add_node("fallback", fallback)
        builder.add_node("refuse", refuse)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "grade_evidence")
        builder.add_conditional_edges(
            "grade_evidence",
            route_evidence,
        )
        builder.add_edge("generate", "verify_answer")
        builder.add_conditional_edges("verify_answer", route_answer)
        builder.add_edge("complete", END)
        builder.add_edge("fallback", END)
        builder.add_edge("refuse", END)
        return builder.compile()


def build_answer_generator_from_env() -> AnswerGenerator:
    provider = os.getenv("FIRMWARE_RAG_GENERATOR", "extractive").lower()
    if provider == "extractive":
        return ExtractiveAnswerGenerator()
    if provider == "ollama":
        return OllamaAnswerGenerator(
            model=os.getenv(
                "FIRMWARE_RAG_OLLAMA_MODEL",
                "llama3.1:8b",
            ),
            base_url=os.getenv(
                "FIRMWARE_RAG_OLLAMA_URL",
                "http://127.0.0.1:11434",
            ),
        )
    raise ValueError(
        "FIRMWARE_RAG_GENERATOR must be extractive or ollama"
    )
