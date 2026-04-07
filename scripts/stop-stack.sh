#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/common.sh
source "$ROOT/scripts/lib/common.sh"

REPO="$(librarian_repo_root "${1:-}")"
if [ "$#" -gt 0 ]; then
  shift
fi
cd "$REPO"
docker compose down "$@"
