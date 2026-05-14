# Builder — uv with Python 3.14 (trixie-slim, multi-arch)
FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim@sha256:8090f78a8865a8fce78ec0bce3177648879148fb22c6dfc4cc0de51f15aea107 AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    uv sync --frozen --no-install-project --no-dev

# Runtime — Python 3.14 (slim-trixie, multi-arch)
FROM python:3.14-slim@sha256:7a500125bc50693f2214e842a621440a1b1b9cbb2188f74ab045d29ed2ea5856 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        tini \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1000 scout \
    && useradd --system --uid 1000 --gid scout --no-create-home --shell /usr/sbin/nologin scout

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder --chown=scout:scout /app/.venv /app/.venv
COPY --chown=scout:scout src/ /app/src/
COPY --chown=scout:scout alembic/ /app/alembic/
COPY --chown=scout:scout alembic.ini /app/alembic.ini
COPY --chown=scout:scout pyproject.toml README.md LICENSE NOTICE /app/

ARG GIT_SHA="unknown"
ARG BUILD_DATE="unknown"
LABEL org.opencontainers.image.source="https://github.com/securityscout/securityscout" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.title="securityscout" \
      org.opencontainers.image.description="Security advisory triage and validation agent"

USER scout:scout

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=2).status==200 else 1)"

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--no-server-header", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]
