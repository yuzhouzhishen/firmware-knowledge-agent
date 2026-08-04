from __future__ import annotations

from pathlib import Path

from firmware_knowledge_agent.corpus import load_corpus


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_corpus_preserves_source_and_section_metadata() -> None:
    chunks = load_corpus(PROJECT_ROOT / "data/sample/sources.json")

    assert len(chunks) >= 6
    assert {chunk.source_id for chunk in chunks} == {
        "freertos-task-delay",
        "esp-idf-nvs",
        "esp-idf-wifi-events",
    }
    assert all(chunk.section for chunk in chunks)
    assert all(chunk.source_url.startswith("https://") for chunk in chunks)
