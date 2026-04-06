#!/usr/bin/env bash
set -euo pipefail

# Start Mongo + Ollama + Librarian (SSE) via docker compose. Run from repo root or pass REPO.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/common.sh
source "$ROOT/scripts/lib/common.sh"

REPO="$(librarian_repo_root "${1:-}")"
cd "$REPO"

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

echo "Pulling embedding model into Ollama (safe to re-run) ..."
docker compose exec -T ollama ollama pull nomic-embed-text || {
  echo "warning: ollama pull failed — run manually: docker compose exec ollama ollama pull nomic-embed-text" >&2
}

echo ""
echo "Stack is up."
echo "  - MongoDB:     mongodb://localhost:27017"
echo "  - Ollama:      http://localhost:11434"
echo "  - Librarian:   http://localhost:8000/sse  (SSE MCP)"
echo ""
echo "HTTP MCP clients: run scripts/mcp-config-http-clients.sh then follow docs/setup/gemini-antigravity.md"
