FROM python:3.13-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev deps, no editable install)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source
COPY main.py ingest.py query.py utils.py ./

# Copy data and pre-built embeddings
COPY data/ ./data/
COPY embeddings/ ./embeddings/

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "main.py"]
