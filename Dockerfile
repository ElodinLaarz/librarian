FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install production deps (no dev extras, no sentence-transformers by default)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source
COPY src/ ./src/

# Install the project itself
RUN uv sync --frozen --no-dev

# Default to SSE transport for containerised deployments
ENV LIBRARIAN_SERVER_TRANSPORT=sse

EXPOSE 8000

ENTRYPOINT ["uv", "run", "python", "-m", "src"]
