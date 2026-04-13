# Cursor — Librarian MCP (stdio)

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (install and on your **PATH** in a normal terminal)
- [jq](https://jqlang.org/) — required by `mcp-config-cursor.sh` (`brew install jq` / `apt install jq`)
- This repo cloned locally
- [Docker](https://docs.docker.com/get-docker/) — for MongoDB and Ollama when using the default `librarian.config.yaml` (local stack)

## First-time setup

### Global MCP (all Cursor workspaces) — recommended

Installs Librarian once so it is available in every project. Writes **`~/.cursor/mcp.json`** (or **`$CURSOR_MCP_JSON`** if set) and **merges** a **`librarian`** entry into existing **`mcpServers`**.

From the repository root:

```bash
uv sync
chmod +x scripts/*.sh scripts/lib/common.sh  # if your checkout has scripts non-executable
./scripts/init-config.sh
./scripts/mcp-config-cursor.sh --global
```

Custom clone path:

```bash
./scripts/mcp-config-cursor.sh --global /path/to/librarian
```

The script sets **`cwd`** to that repo and **`env.LIBRARIAN_CONFIG`** to **`$REPO/librarian.config.yaml`**. The server runs **`uv run python -m src`** from that directory.

**Avoid duplicates:** if you also have **`librarian`** in **`<repo>/.cursor/mcp.json`**, remove it there or delete the project file so Cursor does not register the same server twice.

### Project-local MCP (this repo only)

Writes **`<repo>/.cursor/mcp.json`**:

```bash
./scripts/mcp-config-cursor.sh
./scripts/mcp-config-cursor.sh /path/to/librarian
```

### Full dev stack + HTTP snippets

Brings up Docker and writes HTTP MCP snippets under **`~/.librarian/`**:

```bash
chmod +x scripts/*.sh scripts/lib/common.sh
./scripts/dev-up.sh
./scripts/mcp-config-cursor.sh --global   # or project-local without --global
```

- **`dev-up.sh`** creates **`librarian.config.yaml`** if missing, runs **`start-stack.sh`**, and **`mcp-config-http-clients.sh`**.

Optional secrets/overrides can live in a repo-root **`.env`** (gitignored); see **`.env.example`**. The server command is **`uv run python -m src`** (stdio MCP).

### After moving or renaming the repo

Run **`mcp-config-cursor.sh`** again (with the same **`--global`** or project mode) — **`cwd`** and **`LIBRARIAN_CONFIG`** are absolute paths.

## Agent rules (automatic library integration)

Cursor has no hook system, so integration is instruction-based via a global
rules file. Run the rules installer after setting up the MCP:

```bash
./scripts/hooks-config-cursor.sh
```

This writes **`~/.cursor/rules/librarian.mdc`** with `alwaysApply: true`, which
Cursor loads in every workspace. The rule instructs the agent to:

| Behaviour | When |
| ------------------------------ | ---------------------------------------------------------------------- |
| Search before every new task | Start of each task — before writing code or a plan |
| Augment every user prompt | On each new message from the user |
| Consult library when stuck | After multiple failed attempts at the same problem |
| Ingest newly learned knowledge | Before finishing a response where something non-obvious was discovered |

Custom repo path:

```bash
./scripts/hooks-config-cursor.sh /path/to/librarian
```

Override rules directory:

```bash
CURSOR_RULES_DIR=/path/to/rules ./scripts/hooks-config-cursor.sh
```

Reload the Cursor window after installing (`Ctrl+Shift+P → Developer: Reload Window`).

## Every time you work (backing services)

Cursor spawns the MCP process per session, but **MongoDB** (and **Ollama** for the default embedding settings) must be running for ingest/search to work.

From the repo:

```bash
./scripts/start-stack.sh
```

When finished:

```bash
./scripts/stop-stack.sh
```

## Verify

### 1. CLI (no Cursor)

From the repo root:

```bash
LIBRARIAN_CONFIG="$PWD/librarian.config.yaml" uv run python -c "from src.server import config; print('OK:', config.database.uri)"
```

You should see `OK:` and your database URI. If this fails, fix config/uv before debugging Cursor.

### 2. Cursor UI

1. With a **global** install, any workspace is fine. With a **project-only** **`.cursor/mcp.json`**, open that repo as the folder root.
1. Open **Cursor Settings** → **MCP** (wording may be **Features → MCP** depending on version).
1. Confirm **librarian** is listed and enabled; use **Refresh** / reload if the editor just created the config.
1. In chat/agent, confirm Librarian tools (e.g. **`library_search`**) appear.

## Troubleshooting

- **`jq` not found** when running **`mcp-config-cursor.sh`**: Install jq (see Prerequisites).
- **`uv` not found** inside Cursor’s MCP process (common when Cursor is launched from the GUI and **`uv`** is only on your shell PATH): In **`~/.cursor/mcp.json`** (or your project **`.cursor/mcp.json`**), set **`command`** to the absolute path from `which uv` (e.g. **`/home/you/.local/bin/uv`**), or add a minimal **`PATH`** under **`env`** so **`uv`** resolves.
- **`uv run` fails / module not found**: Run **`uv sync`** once from the repo root.
- **Connection refused to Mongo / server errors on tools**: Start the stack — **`./scripts/start-stack.sh`**. If you use the compose **MongoDB Atlas Local** image from the host, you may need **`directConnection=true`** in the URI — see **`.env.example`**.
- **Embedding errors**: Ensure Ollama has the model:\
  `docker compose exec ollama ollama pull nomic-embed-text`
- **`library_research` / “Web search is not configured”**: The MCP process loads **`Path.cwd()/.env`** when it starts. Your **`mcp-config-cursor.sh`** entry sets **`cwd`** to this repo, so a repo-root **`.env`** is enough — you do **not** have to duplicate keys under **`~/.cursor/mcp.json`** unless you prefer that. For **Tavily**, set **`LIBRARIAN_WEB_SEARCH_PROVIDER=tavily`** and **`LIBRARIAN_WEB_SEARCH_API_KEY=…`**, or **`TAVILY_API_KEY=…`** (see **`src/services/web_search.py`**). Reload the MCP server / window after editing **`.env`**. If it still fails, verify from the repo:\
  `uv run python -c "from pathlib import Path; from src.config import LibrarianConfig; from src.services.web_search import build_web_search_client; c=LibrarianConfig.from_yaml(Path('librarian.config.yaml')); print(build_web_search_client(c).is_available())"`\
  (should print **`True`**).
- **`librarian.config.yaml`**: Edit for your machine; it is not committed (gitignored). **`init-config.sh`** seeds it from **`config/librarian.local.template.yaml`**.
- **Existing MCP servers disappeared**: **`mcp-config-cursor.sh`** merges into **`mcpServers`**; it should keep other keys. If your file was invalid JSON, back it up and re-run the script.

## Windows

Use **WSL2**: install **uv** and **jq** inside WSL, clone the repo on the Linux filesystem, and **open that folder in Cursor** (Linux paths in **`cwd`** / **`LIBRARIAN_CONFIG`**). Opening the repo via **`\\wsl$\...`** from Windows-native Cursor can break **`uv`** resolution and paths — prefer the **WSL** workspace / remote workflow your Cursor build supports.
