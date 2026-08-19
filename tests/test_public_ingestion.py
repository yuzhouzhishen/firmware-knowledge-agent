from __future__ import annotations

import pytest

from firmware_knowledge_agent.public_ingestion import (
    _validate_source_url,
    convert_rst_headings,
    extract_document_markdown,
)


def test_extract_document_markdown_keeps_structure() -> None:
    html = """
    <html>
      <body>
        <nav><p>Navigation should not be indexed.</p></nav>
        <main>
          <h1>NVS</h1>
          <h2>Commit</h2>
          <p>Call nvs_commit after setting values.</p>
          <pre>nvs_commit(handle);</pre>
        </main>
      </body>
    </html>
    """

    markdown = extract_document_markdown(html)

    assert "# NVS" in markdown
    assert "## Commit" in markdown
    assert "nvs_commit(handle);" in markdown
    assert "Navigation" not in markdown


def test_extract_document_markdown_chooses_largest_content_container() -> None:
    html = """
    <html>
      <body>
        <main><p>Loading</p></main>
        <article>
          <h1>Wi-Fi Driver</h1>
          <p>This is the actual documentation body with useful details.</p>
        </article>
      </body>
    </html>
    """

    markdown = extract_document_markdown(html)

    assert "Wi-Fi Driver" in markdown
    assert "actual documentation body" in markdown


def test_extract_document_markdown_keeps_tables_without_duplicate_cells() -> None:
    html = """
    <html>
      <body>
        <main>
          <h1>Acceptance</h1>
          <table>
            <tr><th>Check</th><th>Expected</th></tr>
            <tr>
              <td><p>VBUS</p></td>
              <td><p>Stable output</p></td>
            </tr>
          </table>
        </main>
      </body>
    </html>
    """

    markdown = extract_document_markdown(html)

    assert "| Check | Expected |" in markdown
    assert "| VBUS | Stable output |" in markdown
    assert markdown.count("Stable output") == 1


def test_extract_document_markdown_does_not_duplicate_list_paragraphs() -> None:
    html = """
    <html>
      <body>
        <main>
          <ul><li><p>Inspect the negotiated protocol.</p></li></ul>
        </main>
      </body>
    </html>
    """

    markdown = extract_document_markdown(html)

    assert markdown == "- Inspect the negotiated protocol."


def test_rejects_non_allowlisted_sources() -> None:
    with pytest.raises(ValueError, match="allowlisted"):
        _validate_source_url("https://example.com/private-doc")


def test_rejects_unrelated_raw_github_sources() -> None:
    with pytest.raises(ValueError, match="official repository"):
        _validate_source_url(
            "https://raw.githubusercontent.com/other/repo/main/doc.md"
        )


def test_accepts_official_espressif_documentation_source() -> None:
    _validate_source_url(
        "https://raw.githubusercontent.com/"
        "espressif/esp-idf/v5.5.3/docs/en/api-guides/wifi.rst"
    )


def test_convert_rst_headings_to_markdown() -> None:
    rst = """Wi-Fi Driver
=============

Event Handling
--------------

Preparation
+++++++++++
"""

    markdown = convert_rst_headings(rst)

    assert "# Wi-Fi Driver" in markdown
    assert "## Event Handling" in markdown
    assert "### Preparation" in markdown
