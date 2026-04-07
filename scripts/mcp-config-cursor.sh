#!/usr/bin/env bash
set -euo pipefail

# Write Cursor MCP config for Librarian (stdio). Default: <repo>/.cursor/mcp.json

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/common.sh
source "$ROOT/scripts/lib/common.sh"
librarian_require_jq

REPO_ARG=""
GLOBAL=0
for arg in "$@"; do
  case "$arg" in
  --global) GLOBAL=1 ;;
  *) REPO_ARG="$arg" ;;
  esac
done

REPO="$(librarian_repo_root "${REPO_ARG}")"
STDIO_ENTRY="$(jq -n \
  --arg cwd "$REPO" \
  --arg cfg "$REPO/librarian.config.yaml" \
  '{command: "uv", args: ["--directory", $cwd, "--quiet", "run", "python", "-m", "src"], cwd: $cwd, env: {LIBRARIAN_CONFIG: $cfg, PYTHONUNBUFFERED: "1"}}')"

if [[ "$GLOBAL" -eq 1 ]]; then
  OUT="${CURSOR_MCP_JSON:-$HOME/.cursor/mcp.json}"
  mkdir -p "$(dirname "$OUT")"
else
  OUT="$REPO/.cursor/mcp.json"
  mkdir -p "$REPO/.cursor"
fi

tmp="$(mktemp)"
if [[ -f "$OUT" ]]; then
  jq --argjson lib "$STDIO_ENTRY" \
    '.mcpServers = (.mcpServers // {}) | .mcpServers.librarian = $lib' \
    "$OUT" >"$tmp"
else
  jq -n --argjson lib "$STDIO_ENTRY" '{mcpServers: {librarian: $lib}}' >"$tmp"
fi
mv "$tmp" "$OUT"

echo "Wrote Cursor MCP config: $OUT"
echo "LIBRARIAN_CONFIG is set to $REPO/librarian.config.yaml in the server env block."
echo "Start backing services: $REPO/scripts/start-stack.sh"
