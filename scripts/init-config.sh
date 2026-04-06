#!/usr/bin/env bash
set -euo pipefail

# Create librarian.config.yaml from template if missing (local docker + Ollama defaults).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/common.sh
source "$ROOT/scripts/lib/common.sh"

REPO="$(librarian_repo_root "${1:-}")"
TEMPLATE="$REPO/config/librarian.local.template.yaml"
TARGET="$REPO/librarian.config.yaml"

if [[ ! -f "$TEMPLATE" ]]; then
  echo "error: missing template $TEMPLATE" >&2
  exit 1
fi

if [[ -f "$TARGET" ]]; then
  echo "Keeping existing $TARGET"
  exit 0
fi

cp "$TEMPLATE" "$TARGET"
echo "Created $TARGET (edit or set LIBRARIAN_* env vars to override)"
