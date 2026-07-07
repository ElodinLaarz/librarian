#!/usr/bin/env bash
set -euo pipefail
# Install Librarian hooks into ~/.claude/settings.json.
#
# Adds:
#   - a UserPromptSubmit hook that auto-searches the library on every prompt
#     and injects relevant context before Claude responds;
#   - a PostToolUse (Write|Edit) hook that mirrors Claude Code auto-memory
#     files into the library (replace-by-source). See docs/claude-code-hooks.md.
#
# Usage:
#   ./scripts/hooks-config-claude-code.sh                # default paths
#   ./scripts/hooks-config-claude-code.sh /path/to/repo  # custom repo path

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/common.sh
source "$ROOT/scripts/lib/common.sh"
librarian_require_jq

REPO="$(librarian_repo_root "${1:-}")"
SETTINGS="${CLAUDE_SETTINGS_JSON:-$HOME/.claude/settings.json}"
SEARCH_SCRIPT="$REPO/scripts/librarian-hook.py"
SYNC_SCRIPT="$REPO/scripts/claude-memory-sync-hook.py"

SEARCH_CMD="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --directory $REPO --quiet python $SEARCH_SCRIPT 2>/dev/null"
SYNC_CMD="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --directory $REPO --quiet python $SYNC_SCRIPT"

SEARCH_ENTRY="$(jq -n --arg cmd "$SEARCH_CMD" \
  '{type: "command", command: $cmd, timeout: 30, statusMessage: "Searching librarian library…"}')"
SYNC_ENTRY="$(jq -n --arg cmd "$SYNC_CMD" \
  '{type: "command", command: $cmd, async: true, timeout: 180, statusMessage: "Syncing memory to librarian…"}')"

mkdir -p "$(dirname "$SETTINGS")"
tmp="$(mktemp)"

if [[ ! -f "$SETTINGS" ]]; then
  echo '{}' >"$SETTINGS"
fi

# Merge: add/replace the librarian entries, preserving all other hooks.
jq --argjson search "$SEARCH_ENTRY" --argjson sync "$SYNC_ENTRY" '
  .hooks = (.hooks // {}) |
  .hooks.UserPromptSubmit = (
    (.hooks.UserPromptSubmit // [])
    | map(select(.hooks[0].command | strings | test("librarian-hook") | not))
  ) |
  .hooks.UserPromptSubmit += [{hooks: [$search]}] |
  .hooks.PostToolUse = (
    (.hooks.PostToolUse // [])
    | map(select(.hooks[0].command | strings | test("claude-memory-sync-hook") | not))
  ) |
  .hooks.PostToolUse += [{matcher: "Write|Edit", hooks: [$sync]}]
' "$SETTINGS" >"$tmp"

mv "$tmp" "$SETTINGS"
echo "Librarian hooks installed in: $SETTINGS"
echo "Search hook: $SEARCH_CMD"
echo "Memory-sync hook: $SYNC_CMD"
echo ""
echo "The search hook injects library context on every prompt; the memory-sync"
echo "hook mirrors Claude Code auto-memory files into the library."
echo "Restart Claude Code (or open /hooks once) to pick up the change."
