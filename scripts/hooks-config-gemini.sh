#!/usr/bin/env bash
set -euo pipefail

# Merge Librarian hooks into Gemini CLI settings.json.
# This sets up:
# 1. AfterAgent: Auto-ingest response after every request.
# 2. AfterTool: Auto-ingest when save_memory or library_ingest is called.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/common.sh
source "$ROOT/scripts/lib/common.sh"
librarian_require_jq

REPO="$(librarian_repo_root "${1:-}")"
# Gemini CLI settings location (project-level by default)
SETTINGS="${GEMINI_SETTINGS_JSON:-$HOME/.gemini/settings.json}"
HOOK_SCRIPT="$REPO/scripts/librarian-hook.py"

# Command to run the hook
HOOK_CMD="uv run --directory $REPO --quiet python $HOOK_SCRIPT"

mkdir -p "$(dirname "$SETTINGS")"

# Hook entries for Gemini CLI
AFTER_AGENT_HOOK="$(jq -n --arg cmd "$HOOK_CMD" --arg name "librarian-after-agent" \
  '{matcher: "*", hooks: [{name: $name, type: "command", command: $cmd}]}')"

AFTER_TOOL_HOOK="$(jq -n --arg cmd "$HOOK_CMD" --arg name "librarian-after-tool" \
  '{matcher: "save_memory|library_ingest|mcp_librarian_library_ingest", hooks: [{name: $name, type: "command", command: $cmd}]}')"

tmp="$(mktemp)"

if [[ -f "$SETTINGS" ]]; then
  # Merge into existing settings.json
  jq --argjson agent "$AFTER_AGENT_HOOK" --argjson tool "$AFTER_TOOL_HOOK" '
    .hooks = (.hooks // {}) |
    
    # Update AfterAgent
    .hooks.AfterAgent = (
      (.hooks.AfterAgent // []) 
      | map(select(.hooks[0].name != "librarian-after-agent"))
    ) |
    .hooks.AfterAgent += [$agent] |

    # Update AfterTool
    .hooks.AfterTool = (
      (.hooks.AfterTool // [])
      | map(select(.hooks[0].name != "librarian-after-tool"))
    ) |
    .hooks.AfterTool += [$tool]
  ' "$SETTINGS" >"$tmp"
else
  # Create new settings.json
  jq -n --argjson agent "$AFTER_AGENT_HOOK" --argjson tool "$AFTER_TOOL_HOOK" '{
    hooks: {
      AfterAgent: [$agent],
      AfterTool: [$tool]
    }
  }' >"$tmp"
fi

mv "$tmp" "$SETTINGS"

echo "Librarian Gemini CLI hooks installed in: $SETTINGS"
echo "These hooks will automatically ingest new knowledge from responses and memory tools."
