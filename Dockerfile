# syntax=docker/dockerfile:1

# Mesma base nos dois estágios: mantém Python e bibliotecas de sistema compatíveis.
FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

# Alterar código não invalida a camada de dependências.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/raw

# Instalação não editável: o pacote já está dentro do ambiente virtual.
COPY --from=builder /app/.venv /app/.venv
COPY scripts/ ./scripts/
USER appuser

# O CSV é fornecido no docker run, não incorporado à imagem.
CMD ["python", "scripts/validate_data.py"]
