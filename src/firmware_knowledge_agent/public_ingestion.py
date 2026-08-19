from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/public_sources.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "var/public-corpus"
_ALLOWED_HOSTS = {
    "docs.espressif.com",
    "raw.githubusercontent.com",
}
_ALLOWED_GITHUB_PREFIXES = (
    (
        "https://raw.githubusercontent.com/"
        "FreeRTOS/FreeRTOS-Kernel-Book/"
    ),
    "https://raw.githubusercontent.com/espressif/esp-idf/",
)
_CONTENT_SELECTORS = (
    "main",
    "[role='main']",
    ".document",
    ".wy-nav-content",
    "article",
)


class PublicSource(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]+$")
    title: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    version: str = Field(min_length=1)
    license_note: str = Field(min_length=1)


def ingest_public_sources(
    manifest_path: Path,
    output_dir: Path,
) -> Path:
    sources = _load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog: list[dict[str, str]] = []

    for source in sources:
        _validate_source_url(source.source_url)
        response = requests.get(
            source.source_url,
            timeout=(5, 30),
            headers={
                "User-Agent": (
                    "firmware-knowledge-agent/0.1 "
                    "(local documentation study project)"
                )
            },
        )
        response.raise_for_status()
        response_text = _decode_response(response)
        if response.url.startswith(_ALLOWED_GITHUB_PREFIXES):
            markdown = response_text.strip()
            if urlparse(response.url).path.endswith(".rst"):
                markdown = convert_rst_headings(markdown)
        else:
            markdown = extract_document_markdown(response_text)
        if len(markdown) < 200:
            raise ValueError(
                f"extracted document is unexpectedly short: {source.id}"
            )
        document_name = f"{source.id}.md"
        (output_dir / document_name).write_text(
            markdown,
            encoding="utf-8",
        )
        catalog.append(
            {
                **source.model_dump(),
                "path": document_name,
            }
        )

    catalog_path = output_dir / "sources.json"
    catalog_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return catalog_path


def extract_document_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    candidates = [
        node
        for selector in _CONTENT_SELECTORS
        for node in soup.select(selector)
    ]
    if candidates:
        root = max(
            candidates,
            key=lambda node: len(node.get_text(" ", strip=True)),
        )
    elif soup.body is not None:
        root = soup.body
    else:
        raise ValueError("HTML document has no readable body")

    lines: list[str] = []
    for element in root.find_all(
        ["h1", "h2", "h3", "h4", "p", "pre", "li", "tr"],
        recursive=True,
    ):
        if not isinstance(element, Tag):
            continue
        if _is_nested_block(element, root):
            continue
        if element.name == "tr":
            cells = [
                _normalize_text(cell.get_text(" ", strip=True))
                for cell in element.find_all(["th", "td"], recursive=False)
            ]
            cells = [cell for cell in cells if cell]
            if cells:
                lines.append("| " + " | ".join(cells) + " |")
            continue
        text = _normalize_text(element.get_text(" ", strip=True))
        if not text:
            continue
        if element.name in {"h1", "h2", "h3", "h4"}:
            lines.append(f"{'#' * int(element.name[1])} {text}")
        elif element.name == "pre":
            lines.append(f"```\n{text}\n```")
        elif element.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)
    return "\n\n".join(_deduplicate_adjacent(lines)).strip()


def _is_nested_block(element: Tag, root: Tag) -> bool:
    """Avoid indexing table/list content both as a parent and as children."""
    parent = element.parent
    while isinstance(parent, Tag) and parent is not root:
        if parent.name in {"pre", "li", "tr"}:
            return True
        parent = parent.parent
    return False


def convert_rst_headings(content: str) -> str:
    lines = content.splitlines()
    converted: list[str] = []
    heading_levels = {
        "=": 1,
        "-": 2,
        "+": 3,
        "~": 3,
    }
    index = 0
    while index < len(lines):
        if index + 1 < len(lines):
            title = lines[index].strip()
            underline = lines[index + 1].strip()
            level = heading_levels.get(underline[:1])
            if (
                title
                and level is not None
                and len(underline) >= len(title)
                and len(set(underline)) == 1
            ):
                converted.append(f"{'#' * level} {title}")
                index += 2
                continue
        converted.append(lines[index])
        index += 1
    return "\n".join(converted)


def _load_manifest(path: Path) -> list[PublicSource]:
    raw_sources = json.loads(path.read_text(encoding="utf-8"))
    return [PublicSource.model_validate(item) for item in raw_sources]


def _validate_source_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_HOSTS:
        raise ValueError(
            "public source must use HTTPS on an allowlisted official host"
        )
    if (
        parsed.hostname == "raw.githubusercontent.com"
        and not url.startswith(_ALLOWED_GITHUB_PREFIXES)
    ):
        raise ValueError(
            "GitHub source must belong to an allowlisted official repository"
        )


def _decode_response(response: requests.Response) -> str:
    try:
        return response.content.decode("utf-8")
    except UnicodeDecodeError:
        return response.text


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _deduplicate_adjacent(lines: list[str]) -> list[str]:
    deduplicated: list[str] = []
    for line in lines:
        if not deduplicated or line != deduplicated[-1]:
            deduplicated.append(line)
    return deduplicated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch allowlisted public firmware documentation.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()
    catalog_path = ingest_public_sources(args.manifest, args.output)
    print(f"catalog: {catalog_path}")


if __name__ == "__main__":
    main()
