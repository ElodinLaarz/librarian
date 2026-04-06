# Claude — Librarian MCP (stdio)

Two common surfaces: **Claude Desktop** (GUI app) and **Claude Code** (CLI / IDE integration). Both use a **stdio** MCP server: they run `uv run python -m src` with `cwd` set to this repo.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [uv](https://docs.astral.sh/uv/)
- [jq](https://jqlang.org/)
- This repo cloned locally

## First-time setup

```bash
chmod +x scripts/*.sh scripts/lib/common.sh
./scripts/dev-up.sh
```

Then install the MCP entry for the product you use:

### Claude Desktop (macOS / Linux)

```bash
./scripts/mcp-config-claude-desktop.sh
```

This **merges** a `librarian` entry into:

| OS | File |
| --- | --- |
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

Custom repo path:

```bash
./scripts/mcp-config-claude-desktop.sh /path/to/librarian
```

Restart **Claude Desktop** after changing the file.

### Claude Code

```bash
./scripts/mcp-config-claude-code.sh
```

This **merges** into **`~/.claude.json`** under `mcpServers.librarian` (override with **`CLAUDE_CODE_JSON`** if your build stores MCP elsewhere).

Custom repo path:

```bash
./scripts/mcp-config-claude-code.sh /path/to/librarian
```

If your Claude Code version uses a different config location, open **`~/.claude.json`**, copy the `librarian` object from `mcpServers`, and paste it into the file your app documents.

Each entry includes:

- `command`: `uv`
- `args`: `run`, `python`, `-m`, `src`
- `cwd`: repository root
- `env.LIBRARIAN_CONFIG`: path to **`librarian.config.yaml`**

## Every time you work (backing services)

```bash
cd /path/to/librarian
./scripts/start-stack.sh
```

Stop when done:

```bash
./scripts/stop-stack.sh
```

## Windows (Claude Desktop)

The bash merge script targets macOS/Linux paths. On Windows, open **`%APPDATA%\Claude\claude_desktop_config.json`** and add under **`mcpServers`**:

```json
"librarian": {
  "command": "uv",
  "args": ["run", "python", "-m", "src"],
  "cwd": "C:\\path\\to\\librarian",
  "env": {
    "LIBRARIAN_CONFIG": "C:\\path\\to\\librarian\\librarian.config.yaml"
  }
}
```

Use **WSL** if you prefer the automated scripts.

## Troubleshooting

- **No merge / jq errors**: Install `jq`, or merge the JSON manually using the Cursor example in the main README with the `env` block above.
- **Mongo / Ollama**: Same as [Cursor](cursor.md#troubleshooting).
