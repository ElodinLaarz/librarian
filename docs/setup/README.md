# Librarian setup guides

Scripts assume **bash**, **Docker**, **uv**, and **jq** (`brew install jq` / `apt install jq`).

| Guide | Use case |
| --- | --- |
| [Cursor](cursor.md) | Stdio MCP from Cursor (project **`.cursor/mcp.json`** or global **`~/.cursor/mcp.json`** via `mcp-config-cursor.sh --global`) |
| [Claude](claude.md) | Claude Desktop + Claude Code (stdio) |
| [Gemini / Antigravity](gemini-antigravity.md) | Stdio (via `gemini mcp add`) or HTTP (SSE / streamable-http) |

## Quick commands (repo root)

| Script | Purpose |
| --- | --- |
| `./scripts/dev-up.sh [REPO]` | Create `librarian.config.yaml` if missing, start Docker stack, write HTTP MCP snippets to `~/.librarian/` |
| `./scripts/start-stack.sh [REPO]` | Start Mongo + Ollama + Librarian container (SSE on `:8000`) |
| `./scripts/stop-stack.sh [REPO]` | `docker compose down` |
| `./scripts/init-config.sh [REPO]` | Copy `config/librarian.local.template.yaml` → `librarian.config.yaml` if missing |
| `uv run python scripts/demo_librarian.py` | Offline smoke test: search → ingest → search → research → search (no Docker/API keys) |

Optional first argument **`[REPO]`** is the path to this repository (default: parent of `scripts/`).

`librarian.config.yaml` is gitignored; the template lives at `config/librarian.local.template.yaml`.

## Ollama model setup

The first run may need to pull a large Ollama image/model. Manual commands:

```bash
docker compose up -d ollama
docker compose exec -T ollama ollama pull nomic-embed-text
docker compose exec -T ollama ollama list
```

If startup fails, check:

```bash
docker compose logs --tail=100 ollama
```
