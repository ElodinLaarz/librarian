#!/usr/bin/env bash
set -euo pipefail

# Write MCP snippet for HTTP/SSE clients (Gemini CLI, Antigravity, etc.).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/common.sh
source "$ROOT/scripts/lib/common.sh"
librarian_require_jq

BASE_URL="${LIBRARIAN_SSE_URL:-http://localhost:8000/sse}"
OUT_DIR="${LIBRARIAN_MCP_SNIPPET_DIR:-$HOME/.librarian}"
mkdir -p "$OUT_DIR"

# Full document some clients accept as a file or merge manually
jq -n --arg url "$BASE_URL" '{mcpServers: {librarian: {url: $url}}}' \
  >"$OUT_DIR/mcp-http-librarian.json"

# streamable-http alternative (path may differ per FastMCP version — adjust if needed)
STREAM_URL="${LIBRARIAN_STREAMABLE_HTTP_URL:-http://localhost:8000/mcp}"
jq -n --arg url "$STREAM_URL" '{mcpServers: {librarian_streamable: {url: $url}}}' \
  >"$OUT_DIR/mcp-streamable-librarian.json"

echo "Wrote:"
echo "  $OUT_DIR/mcp-http-librarian.json       (SSE — default)"
echo "  $OUT_DIR/mcp-streamable-librarian.json (streamable-http — try if your client requires it)"
echo ""
echo "Override base URL: LIBRARIAN_SSE_URL=https://host:8000/sse $0"
echo "Start the server: $ROOT/scripts/start-stack.sh"
