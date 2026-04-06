#!/usr/bin/env bash
# Shared helpers — source from scripts in scripts/ (not scripts/lib/).

librarian_require_jq() {
  if ! command -v jq >/dev/null 2>&1; then
    echo "error: jq is required. Install: brew install jq  |  apt install jq" >&2
    exit 1
  fi
}

# Claude Desktop JSON path (macOS / Linux). Empty if unknown OS.
librarian_claude_desktop_config_path() {
  case "$(uname -s)" in
  Darwin) echo "$HOME/Library/Application Support/Claude/claude_desktop_config.json" ;;
  Linux) echo "${XDG_CONFIG_HOME:-$HOME/.config}/Claude/claude_desktop_config.json" ;;
  *) echo "" ;;
  esac
}

# Repo root: optional first argument, else parent of the directory containing this script.
librarian_repo_root() {
  if [[ -n "${1:-}" ]]; then
    cd "$1" && pwd
    return
  fi
  # This file lives in scripts/lib/ → two levels up is repo root.
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}
