#!/usr/bin/env bash
set -euo pipefail
# One-shot installer: set up Librarian agent hooks for all supported tools.
#
# Installs:
#   Claude Code  — UserPromptSubmit hook in ~/.claude/settings.json
#   Cursor       — Always-applied global rule in ~/.cursor/rules/librarian.mdc
#   Antigravity  — Instructions appended to ~/.gemini/GEMINI.md
#
# Usage:
#   ./scripts/hooks-setup.sh                # auto-detect repo root
#   ./scripts/hooks-setup.sh /path/to/repo  # explicit repo path

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${1:-$ROOT}"

echo "==> Installing Librarian hooks for Claude Code…"
"$ROOT/scripts/hooks-config-claude-code.sh" "$REPO"

echo ""
echo "==> Installing Librarian rules for Cursor…"
"$ROOT/scripts/hooks-config-cursor.sh" "$REPO"

echo ""
echo "==> Installing Librarian instructions for Antigravity…"
"$ROOT/scripts/hooks-config-antigravity.sh"

echo ""
echo "==> Installing Librarian hooks for Gemini CLI…"
"$ROOT/scripts/hooks-config-gemini.sh" "$REPO"

echo ""
echo "All hooks installed. Summary:"
echo "  Claude Code  ~/.claude/settings.json  (UserPromptSubmit + PostToolUse hooks)"
echo "  Cursor       ~/.cursor/rules/librarian.mdc"
echo "  Antigravity  ~/.gemini/GEMINI.md"
echo "  Gemini CLI   ~/.gemini/settings.json (AfterAgent/AfterTool hooks)"
