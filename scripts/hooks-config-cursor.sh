#!/usr/bin/env bash
set -euo pipefail
# Install Librarian agent rules for Cursor as a global rules file.
#
# Writes ~/.cursor/rules/librarian.mdc — Cursor loads this automatically
# in every workspace when "Always" is set in the rule frontmatter.
#
# Usage:
#   ./scripts/hooks-config-cursor.sh                # default paths
#   ./scripts/hooks-config-cursor.sh /path/to/repo  # custom repo path

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/common.sh
source "$ROOT/scripts/lib/common.sh"

REPO="$(librarian_repo_root "${1:-}")"
RULES_DIR="${CURSOR_RULES_DIR:-$HOME/.cursor/rules}"
RULES_FILE="$RULES_DIR/librarian.mdc"

mkdir -p "$RULES_DIR"

cat >"$RULES_FILE" <<EOF
---
description: Librarian MCP — automatic knowledge-base integration
globs:
alwaysApply: true
---

# Librarian Knowledge Base

You have access to a personal knowledge library via the \`librarian\` MCP server
(\`library_search\`, \`library_ingest\`, \`library_research\` tools).

## When to use the library

1. **Start of every task** — Before writing any code or making a plan, call
   \`library_search\` with a short description of the task.  Use whatever results
   are returned to inform your approach.

2. **Every user message** — On each new prompt, call \`library_search\` with the
   user's request as the query.  Surface any relevant Tomes in your thinking
   before you respond.

3. **When stuck** — If you have tried multiple approaches and are still blocked,
   call \`library_search\` describing what you are trying to do and what has
   failed.  The library may contain a known fix or a relevant prior note.

4. **After learning something new** — If you discover something non-obvious
   (an undocumented behaviour, a non-obvious fix, a useful pattern, a gotcha),
   call \`library_ingest\` with a clear, self-contained explanation before
   finishing your response.  Write it as if future-you has no context.

## Format guidance

- Keep \`library_search\` queries short and topic-focused (< 200 chars).
- Pass \`skip_verify: true\` to \`library_ingest\` for factual/technical notes.
- Always call these tools silently — do not narrate the lookup to the user
  unless the results are directly relevant to share.
EOF

echo "Cursor rules installed: $RULES_FILE"
echo "Restart Cursor (or reload the window) for the rule to take effect."
