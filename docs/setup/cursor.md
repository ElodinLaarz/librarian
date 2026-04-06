# Cursor — Librarian MCP (stdio)

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [uv](https://docs.astral.sh/uv/)
- [jq](https://jqlang.org/) for setup scripts
- This repo cloned locally

## First-time setup

From the repository root (or pass the repo path as the first argument to each script):

```bash
chmod +x scripts/*.sh scripts/lib/common.sh
./scripts/dev-up.sh
./scripts/mcp-config-cursor.sh
```

- **`dev-up.sh`** creates **`librarian.config.yaml`** (from the template) if it does not exist, starts **Mongo**, **Ollama**, and the **Librarian** container (for HTTP use), and writes HTTP MCP snippets under **`~/.librarian/`**.
- **`mcp-config-cursor.sh`** writes **`.cursor/mcp.json`** in the repo with a `librarian` server entry. It sets **`cwd`** to the repo and **`env.LIBRARIAN_CONFIG`** to **`$REPO/librarian.config.yaml`**.

### Project vs global Cursor config

- Default: **`<repo>/.cursor/mcp.json`** (recommended for a single project).
- Global: `./scripts/mcp-config-cursor.sh --global` writes **`~/.cursor/mcp.json`** (or **`$CURSOR_MCP_JSON`** if set).

### Custom repo path

```bash
./scripts/mcp-config-cursor.sh /path/to/librarian
./scripts/mcp-config-cursor.sh --global /path/to/librarian
```

## Every time you work (backing services)

Cursor spawns the MCP process per session, but **MongoDB and Ollama** must be running for ingest/search to work.

From the repo:

```bash
./scripts/start-stack.sh
```

When finished:

```bash
./scripts/stop-stack.sh
```

## Verify

1. Restart Cursor (or reload MCP servers).
1. Confirm **Librarian** appears in MCP tools.
1. Run a **`library_search`** from the agent.

## Troubleshooting

- **`uv` not found**: Install uv or change `command`/`args` in `.cursor/mcp.json` to your Python/venv.
- **Connection refused to Mongo**: Run `./scripts/start-stack.sh`.
- **Embedding errors**: Ensure Ollama has the model:\
  `docker compose exec ollama ollama pull nomic-embed-text`
- **`librarian.config.yaml`**: Edit for your machine; it is not committed (gitignored).

## Windows

Use **WSL2** (same bash scripts) or translate paths: repo on Windows should use WSL paths in `cwd` if Cursor uses the WSL extension.
