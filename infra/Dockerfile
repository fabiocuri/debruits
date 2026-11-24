FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/   /app/src/

WORKDIR /app/src

EXPOSE 5000

CMD ["/app/.venv/bin/gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "180", "api:create_app()"]
