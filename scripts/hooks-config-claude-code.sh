#!/usr/bin/env bash
set -euo pipefail
# Install Librarian hooks into ~/.claude/settings.json.
#
# Adds a UserPromptSubmit hook that auto-searches the library on every prompt
# and injects relevant context before Claude responds.
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
HOOK_SCRIPT="$REPO/scripts/librarian-hook.py"

UVX_CMD="uv run --directory $REPO --quiet python $HOOK_SCRIPT"

HOOK_ENTRY="$(jq -n --arg cmd "$UVX_CMD" '{type: "command", command: $cmd}')"

mkdir -p "$(dirname "$SETTINGS")"
tmp="$(mktemp)"

if [[ -f "$SETTINGS" ]]; then
  # Merge: add/replace librarian UserPromptSubmit hook, preserving all others.
  jq --argjson hook "$HOOK_ENTRY" '
    .hooks = (.hooks // {}) |
    # Remove any existing librarian-hook entry from UserPromptSubmit
    .hooks.UserPromptSubmit = (
      (.hooks.UserPromptSubmit // [])
      | map(select(.hooks[0].command | strings | test("librarian-hook") | not))
    ) |
    # Append the new hook
    .hooks.UserPromptSubmit += [{hooks: [$hook]}]
  ' "$SETTINGS" >"$tmp"
else
  jq -n --argjson hook "$HOOK_ENTRY" '{
    hooks: {
      UserPromptSubmit: [{hooks: [$hook]}]
    }
  }' >"$tmp"
fi

mv "$tmp" "$SETTINGS"
echo "Librarian hook installed in: $SETTINGS"
echo "Hook command: $UVX_CMD"
echo ""
echo "The hook will auto-search the library on every prompt and inject"
echo "relevant context. Restart Claude Code to pick up the change."
