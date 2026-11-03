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
    && mkdir -p /app/.dvc /app/artifacts /app/data/interim /app/data/raw \
        /app/models /app/reports/generated \
    && chown -R appuser:appuser /app

# A instalação não editável do pacote já está dentro do ambiente virtual.
COPY --from=builder /app/.venv /app/.venv

# Fontes e arquivos necessários para `dvc repro`. O CSV continua vindo por bind mount.
COPY --from=builder --chown=appuser:appuser /app/src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
COPY --chown=appuser:appuser dvc.yaml dvc.lock params.yaml ./
COPY --chown=appuser:appuser .dvc/ ./.dvc/
COPY --chown=appuser:appuser data/raw/online_shoppers_intention.csv.dvc ./data/raw/

# A imagem não inclui o histórico Git; DVC opera em modo sem SCM no container.
RUN sed -i '/^\[core\]$/a\    no_scm = true' .dvc/config

USER appuser

# O CSV é fornecido no docker run em /app/data/raw.
CMD ["dvc", "repro"]
