from __future__ import annotations

import json
from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from firmware_knowledge_agent.models import KnowledgeChunk, SourceSpec


def load_corpus(
    catalog_path: Path,
    *,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> list[KnowledgeChunk]:
    catalog_path = catalog_path.resolve()
    root = catalog_path.parent
    raw_sources = json.loads(catalog_path.read_text(encoding="utf-8"))
    sources = [SourceSpec.model_validate(item) for item in raw_sources]
    chunks: list[KnowledgeChunk] = []

    for source in sources:
        document_path = (root / source.path).resolve()
        if not document_path.is_relative_to(root):
            raise ValueError(f"source path leaves catalog directory: {source.path}")
        content = document_path.read_text(encoding="utf-8")
        chunks.extend(
            _split_source(
                source,
                content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return chunks


def _split_source(
    source: SourceSpec,
    content: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[KnowledgeChunk]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,
    )
    length_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ". ", " ", ""],
    )
    documents = length_splitter.split_documents(
        header_splitter.split_text(content)
    )

    chunks: list[KnowledgeChunk] = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        section = (
            metadata.get("h3")
            or metadata.get("h2")
            or metadata.get("h1")
            or source.title
        )
        chunks.append(
            KnowledgeChunk(
                id=f"{source.id}:{index:04d}",
                source_id=source.id,
                title=source.title,
                section=str(section),
                content=document.page_content.strip(),
                source_url=source.source_url,
                version=source.version,
                source_type=source.source_type,
                component=source.component,
                confidentiality=source.confidentiality,
            )
        )
    return chunks
