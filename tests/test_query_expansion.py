from __future__ import annotations

from firmware_knowledge_agent.retrieval import expand_firmware_query


def test_mqtt_query_expands_to_network_ready_event() -> None:
    expanded = expand_firmware_query("连接 AP 后能启动 MQTT 吗？")

    assert "IP_EVENT_STA_GOT_IP" in expanded
    assert "socket" in expanded


def test_reconnect_query_expands_to_wifi_event_identifiers() -> None:
    expanded = expand_firmware_query("断线重连应该怎么处理？")

    assert "WIFI_EVENT_STA_DISCONNECTED" in expanded
    assert "esp_wifi_connect" in expanded
    assert "esp_wifi_disconnect" in expanded


def test_task_name_query_expands_to_document_terms() -> None:
    expanded = expand_firmware_query(
        "xTaskCreate 的 pcName 任务名有什么用途？"
    )

    assert "debugging aid" in expanded
    assert "human-readable name" in expanded


def test_nvs_power_loss_query_expands_to_robustness_terms() -> None:
    expanded = expand_firmware_query(
        "写 NVS 键值对时突然断电会丢什么？"
    )

    assert "power off" in expanded
    assert "new key-value pair" in expanded
    assert "being written" in expanded


def test_unknown_query_is_not_modified() -> None:
    query = "PostgreSQL 索引怎么设计？"

    assert expand_firmware_query(query) == query
