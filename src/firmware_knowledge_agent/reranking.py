from __future__ import annotations

from pathlib import Path
from typing import Protocol

from firmware_knowledge_agent.models import SearchHit
from firmware_knowledge_agent.retrieval import Retriever


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        hits: list[SearchHit],
        *,
        top_k: int,
    ) -> list[SearchHit]: ...


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = (
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        ),
        *,
        model_file: str = "onnx/model_quint8_avx2.onnx",
        cache_dir: Path = Path("var/models"),
    ) -> None:
        try:
            import onnxruntime as ort
            from huggingface_hub import (
                hf_hub_download,
            )
            from huggingface_hub.errors import LocalEntryNotFoundError
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "ONNX CrossEncoder reranking requires the 'rerank' extra: "
                "uv sync --extra dev --extra rerank"
            ) from exc
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            model_path = hf_hub_download(
                repo_id=model_name,
                filename=model_file,
                cache_dir=str(cache_dir),
                local_files_only=True,
            )
        except LocalEntryNotFoundError:
            model_path = hf_hub_download(
                repo_id=model_name,
                filename=model_file,
                cache_dir=str(cache_dir),
            )
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                local_files_only=True,
            )
        except OSError:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
            )
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {
            item.name for item in self._session.get_inputs()
        }

    def rerank(
        self,
        query: str,
        hits: list[SearchHit],
        *,
        top_k: int,
    ) -> list[SearchHit]:
        if not hits:
            return []
        passages = [
            "\n".join(
                (
                    hit.chunk.title,
                    hit.chunk.section,
                    hit.chunk.content,
                )
            )
            for hit in hits
        ]
        encoded = self._tokenizer(
            [query] * len(passages),
            passages,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        scores = self._session.run(
            None,
            {
                name: encoded[name]
                for name in self._input_names
                if name in encoded
            },
        )[0].reshape(-1)
        ordered = sorted(
            zip(hits, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:top_k]
        return [
            SearchHit(
                rank=rank,
                score=round(float(score), 6),
                chunk=hit.chunk,
            )
            for rank, (hit, score) in enumerate(ordered, start=1)
        ]


class RerankingRetriever:
    def __init__(
        self,
        inner: Retriever,
        reranker: Reranker,
        *,
        candidate_count: int = 12,
    ) -> None:
        if candidate_count <= 0:
            raise ValueError("reranker candidate_count must be positive")
        self._inner = inner
        self._reranker = reranker
        self._candidate_count = candidate_count

    def search(self, query: str, *, top_k: int = 3) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        candidates = self._inner.search(
            query,
            top_k=max(top_k, self._candidate_count),
        )
        return self._reranker.rerank(
            query,
            candidates,
            top_k=top_k,
        )

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()
