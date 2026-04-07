#!/usr/bin/env bash
set -euo pipefail

# One-shot local dev: config file + docker stack + HTTP MCP snippet files.

if [[ -n "${1:-}" ]]; then
  REPO="$(cd "$1" && pwd)"
else
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

"$REPO/scripts/init-config.sh" "$REPO"
"$REPO/scripts/start-stack.sh" "$REPO"
"$REPO/scripts/mcp-config-http-clients.sh"

echo ""
echo "Optional: configure editors —"
echo "  Cursor:         $REPO/scripts/mcp-config-cursor.sh $REPO"
echo "  Claude Desktop: $REPO/scripts/mcp-config-claude-desktop.sh $REPO"
echo "  Claude Code:    $REPO/scripts/mcp-config-claude-code.sh $REPO"
