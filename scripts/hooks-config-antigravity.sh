#!/usr/bin/env bash
set -euo pipefail
# Append Librarian usage instructions to ~/.gemini/GEMINI.md.
#
# Gemini CLI / Antigravity reads GEMINI.md as persistent agent instructions.
# This script adds a section that teaches the agent when and how to use the
# Librarian MCP tools.
#
# Usage:
#   ./scripts/hooks-config-antigravity.sh   # appends to ~/.gemini/GEMINI.md

GEMINI_MD="${GEMINI_MD:-$HOME/.gemini/GEMINI.md}"

# Guard: do not append a duplicate section.
if grep -q "## Librarian Knowledge Base" "$GEMINI_MD" 2>/dev/null; then
  echo "Librarian section already present in: $GEMINI_MD"
  echo "Edit it manually if you want to update it."
  exit 0
fi

mkdir -p "$(dirname "$GEMINI_MD")"

cat >>"$GEMINI_MD" <<'EOF'

## Librarian Knowledge Base

You have access to a personal knowledge library via the `librarian` MCP server
(`library_search`, `library_ingest`, `library_research` tools).

### When to use the library

1. **Start of every task** — Before writing any code or making a plan, call
   `library_search` with a short description of the task.  Use whatever results
   are returned to inform your approach.

2. **Every user message** — On each new prompt, call `library_search` with the
   user's request as the query.  Surface any relevant Tomes in your thinking
   before you respond.

3. **When stuck** — If you have tried multiple approaches and are still blocked,
   call `library_search` describing what you are trying to do and what has
   failed.  The library may contain a known fix or a relevant prior note.

4. **After learning something new** — If you discover something non-obvious
   (an undocumented behaviour, a non-obvious fix, a useful pattern, a gotcha),
   call `library_ingest` with a clear, self-contained explanation before
   finishing your response.  Write it as if future-you has no context.

### Format guidance

- Keep `library_search` queries short and topic-focused (< 200 chars).
- Pass `skip_verify: true` to `library_ingest` for factual/technical notes.
- Call these tools silently — do not narrate the lookup to the user unless the
  results are directly relevant to share.
EOF

echo "Librarian instructions appended to: $GEMINI_MD"
