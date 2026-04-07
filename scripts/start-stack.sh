#!/usr/bin/env bash
set -euo pipefail

# Start Mongo + Ollama + Librarian (SSE) via docker compose. Run from repo root or pass REPO.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/common.sh
source "$ROOT/scripts/lib/common.sh"

REPO="$(librarian_repo_root "${1:-}")"
cd "$REPO"
MODEL="${LIBRARIAN_OLLAMA_MODEL:-nomic-embed-text}"

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker is required for start-stack." >&2
  echo "Install Docker Desktop/Engine and retry." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "error: cannot reach Docker daemon." >&2
  echo "Start Docker Desktop/Engine, then retry: $REPO/scripts/start-stack.sh" >&2
  exit 1
fi

if [[ ! -f "$REPO/librarian.config.yaml" ]]; then
  echo "No librarian.config.yaml — running init-config.sh first."
  "$REPO/scripts/init-config.sh" "$REPO"
fi

echo "Starting docker compose stack in $REPO ..."
if ! docker compose up -d --wait 2>/dev/null; then
  docker compose up -d
  echo "Waiting for healthchecks (no compose --wait support) ..."
  sleep 25
fi

echo "Waiting for Ollama API to become ready ..."
for _ in $(seq 1 60); do
  if docker compose exec -T ollama ollama list >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! docker compose exec -T ollama ollama list >/dev/null 2>&1; then
  echo "error: Ollama did not become ready within 120s." >&2
  echo "Inspect logs: docker compose logs --tail=100 ollama" >&2
  exit 1
fi

if docker compose exec -T ollama ollama list | grep -q "^${MODEL}"; then
  echo "Ollama model already present: ${MODEL}"
else
  echo "Pulling Ollama model '${MODEL}' (first run can take several minutes) ..."
  if ! docker compose exec -T ollama ollama pull "${MODEL}"; then
    echo "error: failed to pull Ollama model '${MODEL}'." >&2
    echo "Retry manually: docker compose exec -T ollama ollama pull ${MODEL}" >&2
    echo "Then verify:      docker compose exec -T ollama ollama list" >&2
    exit 1
  fi
fi

if ! docker compose exec -T ollama ollama list | grep -q "^${MODEL}"; then
  echo "error: model '${MODEL}' is still missing after pull." >&2
  echo "Inspect logs: docker compose logs --tail=100 ollama" >&2
  exit 1
fi

echo ""
echo "Stack is up."
echo "  - MongoDB:     mongodb://localhost:27017"
echo "  - Ollama:      http://localhost:11434"
echo "  - Librarian:   http://localhost:8000/sse  (SSE MCP)"
echo ""
echo "HTTP MCP clients: run scripts/mcp-config-http-clients.sh then follow docs/setup/gemini-antigravity.md"
