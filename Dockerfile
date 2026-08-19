FROM ghcr.io/astral-sh/uv:0.11.2 AS uv
FROM python:3.13-slim-bookworm

COPY --from=uv /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY data ./data
COPY evals ./evals
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8011

HEALTHCHECK --interval=10s --timeout=5s --start-period=90s --retries=12 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8011/health', timeout=3)"

CMD ["uvicorn", "firmware_knowledge_agent.api:app", "--host", "0.0.0.0", "--port", "8011"]
