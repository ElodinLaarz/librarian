# Librarian setup guides

Scripts assume **bash**, **Docker**, **uv**, and **jq** (`brew install jq` / `apt install jq`).

| Guide | Use case |
| --- | --- |
| [Cursor](cursor.md) | Stdio MCP from Cursor |
| [Claude](claude.md) | Claude Desktop + Claude Code (stdio) |
| [Gemini / Antigravity](gemini-antigravity.md) | HTTP MCP (SSE / streamable-http) |

## Quick commands (repo root)

| Script | Purpose |
| --- | --- |
| `./scripts/dev-up.sh [REPO]` | Create `librarian.config.yaml` if missing, start Docker stack, write HTTP MCP snippets to `~/.librarian/` |
| `./scripts/start-stack.sh [REPO]` | Start Mongo + Ollama + Librarian container (SSE on `:8000`) |
| `./scripts/stop-stack.sh [REPO]` | `docker compose down` |
| `./scripts/init-config.sh [REPO]` | Copy `config/librarian.local.template.yaml` → `librarian.config.yaml` if missing |

Optional first argument **`[REPO]`** is the path to this repository (default: parent of `scripts/`).

`librarian.config.yaml` is gitignored; the template lives at `config/librarian.local.template.yaml`.
