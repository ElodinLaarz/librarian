#!/usr/bin/env bash
set -euo pipefail

# Merge Librarian into Claude Code user config ~/.claude.json (mcpServers).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/common.sh
source "$ROOT/scripts/lib/common.sh"
librarian_require_jq

REPO="$(librarian_repo_root "${1:-}")"
CFG="${CLAUDE_CODE_JSON:-$HOME/.claude.json}"

STDIO_ENTRY="$(jq -n \
  --arg cwd "$REPO" \
  --arg cfg "$REPO/librarian.config.yaml" \
  '{command: "uv", args: ["run", "python", "-m", "src"], cwd: $cwd, env: {LIBRARIAN_CONFIG: $cfg}}')"

mkdir -p "$(dirname "$CFG")"
tmp="$(mktemp)"
if [[ -f "$CFG" ]]; then
  jq --argjson lib "$STDIO_ENTRY" \
    '.mcpServers = (.mcpServers // {}) | .mcpServers.librarian = $lib' \
    "$CFG" >"$tmp"
else
  jq -n --argjson lib "$STDIO_ENTRY" '{mcpServers: {librarian: $lib}}' >"$tmp"
fi
mv "$tmp" "$CFG"

echo "Merged Librarian into: $CFG"
echo "If your Claude Code build uses a different file, copy the librarian stanza from this file or set CLAUDE_CODE_JSON."
echo "Restart Claude Code. Backing services: $REPO/scripts/start-stack.sh"
