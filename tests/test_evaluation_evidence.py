from __future__ import annotations

from firmware_knowledge_agent.evaluation import _contains_expected_evidence


def test_evidence_requires_one_term_from_every_group() -> None:
    groups = [
        ["nvs_commit"],
        ["not be updated", "持久化"],
    ]

    assert _contains_expected_evidence(
        "Storage will not be updated until nvs_commit is called.",
        groups,
    )
    assert not _contains_expected_evidence(
        "The NVS API includes nvs_commit.",
        groups,
    )
