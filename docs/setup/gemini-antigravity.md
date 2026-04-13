# Gemini CLI / Antigravity — Librarian MCP Setup

Gemini CLI and Antigravity connect to the Librarian via **`stdio`** transport — no Docker or HTTP server needed. The MCP process is launched automatically using `uv run`.

## Prerequisites

| Requirement | Notes |
| -------------------------------- | ------------------------- |
| [uv](https://docs.astral.sh/uv/) | Python package manager |
| [Ollama](https://ollama.com) | Local embedding inference |
| `nomic-embed-text` model | Pulled via `ollama pull` |

## First-time setup

```bash
# 1. Clone and install dependencies
git clone https://github.com/ElodinLaarz/librarian
cd librarian
uv sync

# 2. Pull the embedding model
ollama pull nomic-embed-text

# 3. Initialize config (creates librarian.config.yaml)
./scripts/init-config.sh
```

## Option A — Gemini CLI (recommended)

Use the built-in `mcp add` command to register the librarian. This automatically updates your `.gemini/settings.json` (project) or `~/.gemini/settings.json` (user).

### Register the MCP

```bash
# In the librarian repo root:
gemini mcp add librarian uv -e LIBRARIAN_CONFIG=$(pwd)/librarian.config.yaml -- run python -m src
```

*Note: Use `--scope user` if you want it to be available in all projects.*

## Option B — Antigravity / Manual Registration

Register the MCP in Antigravity or manually edit your `~/.gemini/settings.json` or `~/.gemini/antigravity/mcp_config.json`:

```json
{
  "mcpServers": {
    "librarian": {
      "command": "uv",
      "args": ["run", "python", "-m", "src"],
      "env": {
        "LIBRARIAN_CONFIG": "/path/to/librarian/librarian.config.yaml"
      },
      "cwd": "/path/to/librarian"
    }
  }
}
```

Replace `/path/to/librarian` with the actual repo path (e.g. `/home/elodin/github/librarian`).

______________________________________________________________________

## Option C — Docker + HTTP (SSE/streamable-http)

> Use this if you want MongoDB vector search, a shared server, or remote access.

The Docker Compose **`librarian`** service sets **`LIBRARIAN_SERVER_TRANSPORT=sse`** and listens on port **8000**.

### Setup

```bash
chmod +x scripts/*.sh scripts/lib/common.sh
./scripts/dev-up.sh
./scripts/mcp-config-http-clients.sh
```

`mcp-config-http-clients.sh` writes connection snippets to `~/.librarian/`.

### Point Antigravity at the HTTP server

Merge the `mcpServers` entry from `~/.librarian/mcp-http-librarian.json` into your config, or use the URL directly: **`http://localhost:8000/sse`**.

______________________________________________________________________

## Agent instructions (automatic library integration)

Antigravity and Gemini CLI read **`~/.gemini/GEMINI.md`** as persistent agent instructions.
Run the installer after setting up the MCP:

```bash
./scripts/hooks-config-antigravity.sh
```

This appends a **Librarian Knowledge Base** section to `~/.gemini/GEMINI.md` that instructs the agent to search, research, and ingest tomes automatically.

______________________________________________________________________

## Troubleshooting

| Symptom | Fix |
| ------------------------- | ------------------------------------------------------------------------- |
| `Missing required config` | `LIBRARIAN_CONFIG` env var is not set or points to wrong file |
| Ollama embedding error | Ensure Ollama is running (`ollama serve`) and model is pulled |
| Connection reset | Confirm `LIBRARIAN_SERVER_TRANSPORT=sse` (not `stdio`) for HTTP processes |
