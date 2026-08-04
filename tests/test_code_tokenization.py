from __future__ import annotations

from firmware_knowledge_agent.retrieval import tokenize


def test_code_identifier_keeps_full_name_and_components() -> None:
    tokens = tokenize("Call nvs_commit after WIFI_EVENT_STA_DISCONNECTED")

    assert "nvs_commit" in tokens
    assert "nvs" in tokens
    assert "commit" in tokens
    assert "wifi_event_sta_disconnected" in tokens
    assert "disconnected" in tokens
