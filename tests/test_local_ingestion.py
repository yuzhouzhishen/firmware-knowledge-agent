from __future__ import annotations

import json
from pathlib import Path

from firmware_knowledge_agent.local_ingestion import (
    ingest_document_bytes,
)


def test_ingest_markdown_creates_private_catalog(tmp_path: Path) -> None:
    catalog_path = tmp_path / "corpus/sources.json"

    result = ingest_document_bytes(
        filename="mqtt-guide.md",
        content=(
            b"# MQTT Guide\n\n"
            b"Reconnect the client with exponential backoff after disconnect. "
            b"Keep retry counters and reset them after a stable connection."
        ),
        catalog_path=catalog_path,
        component="mqtt",
    )

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    document = Path(result.document_path).read_text(encoding="utf-8")
    assert result.created is True
    assert catalog[0]["id"] == "mqtt-guide"
    assert catalog[0]["component"] == "mqtt"
    assert "exponential backoff" in document


def test_ingest_same_source_updates_catalog_without_duplicate(
    tmp_path: Path,
) -> None:
    catalog_path = tmp_path / "corpus/sources.json"
    first = (
        b"# Port Manager\n\n"
        b"The manager owns port state and emits state change events. "
        b"Consumers subscribe to changes instead of polling hardware."
    )
    second = (
        b"# Port Manager\n\n"
        b"The manager validates port transitions and serializes commands. "
        b"Consumers receive a stable state snapshot after each operation."
    )

    ingest_document_bytes(
        filename="port-manager.md",
        content=first,
        catalog_path=catalog_path,
    )
    result = ingest_document_bytes(
        filename="port-manager.md",
        content=second,
        catalog_path=catalog_path,
    )

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    assert result.created is False
    assert len(catalog) == 1
    assert "serializes commands" in Path(
        result.document_path
    ).read_text(encoding="utf-8")


def test_ingest_html_extracts_body_and_redacts_sensitive_values(
    tmp_path: Path,
) -> None:
    html = b"""
    <html><body><nav>Ignore navigation</nav><main>
      <h1>Device Setup</h1>
      <p>Configure MQTT before device startup and verify connectivity.</p>
      <p>password=plain-text-secret</p>
      <p>Host: 10.0.0.8</p>
    </main></body></html>
    """

    result = ingest_document_bytes(
        filename="device.html",
        content=html,
        catalog_path=tmp_path / "corpus/sources.json",
    )

    document = Path(result.document_path).read_text(encoding="utf-8")
    assert "Ignore navigation" not in document
    assert "plain-text-secret" not in document
    assert "10.0.0.8" not in document
    assert "<REDACTED_VALUE>" in document
    assert "<REDACTED_IP>" in document
