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

#### Option A — Global (recommended)

```bash
./scripts/mcp-config-claude-code.sh
```

This **merges** a `librarian` entry into **`~/.claude.json`** under `mcpServers.librarian`, making Librarian available in all Claude Code sessions regardless of project.

Custom repo path:

```bash
./scripts/mcp-config-claude-code.sh /path/to/librarian
```

Override the target config file:

```bash
CLAUDE_CODE_JSON=/path/to/config.json ./scripts/mcp-config-claude-code.sh
```

#### Option B — Project-level (`.mcp.json`)

Claude Code also supports a per-project `.mcp.json` in the repo root. This is picked up automatically when you open the project, with no global config changes. Create or edit `.mcp.json`:

```json
{
  "mcpServers": {
    "librarian": {
      "command": "uv",
      "args": ["run", "python", "-m", "src"],
      "cwd": "/path/to/librarian",
      "env": {
        "LIBRARIAN_CONFIG": "/path/to/librarian/librarian.config.yaml"
      }
    }
  }
}
```

> `.mcp.json` is gitignored in this repo — it's personal/machine-specific. Each developer creates their own.

Each entry includes:

- `command`: `uv`
- `args`: `run`, `python`, `-m`, `src`
- `cwd`: repository root
- `env.LIBRARIAN_CONFIG`: path to **`librarian.config.yaml`**

## Agent hooks (automatic library integration)

Installing the MCP server makes the tools _available_; hooks make the agent
_use_ them automatically. Run the hooks installer after setting up the MCP:

```bash
./scripts/hooks-config-claude-code.sh
```

This merges a `UserPromptSubmit` hook into **`~/.claude/settings.json`**. On
every prompt the hook runs `scripts/librarian-hook.py`, which searches the
library and injects any relevant Tomes as `<librarian_context>` before Claude
responds. This covers:

| Behaviour | Mechanism |
| --- | --- |
| Search before every new task | `UserPromptSubmit` hook injects context automatically |
| Augment every user prompt | Same hook — fires on every message |
| Consult library when stuck | Global `~/.claude/CLAUDE.md` instructs the agent to call `library_search` when blocked |
| Ingest newly learned knowledge | Same global `CLAUDE.md` — agent calls `library_ingest` before finishing when it discovers something non-obvious |

The `~/.claude/CLAUDE.md` global instructions file is created by the same
script and persists across all Claude Code sessions.

Custom repo path:

```bash
./scripts/hooks-config-claude-code.sh /path/to/librarian
```

Override the settings file:

```bash
CLAUDE_SETTINGS_JSON=/path/to/settings.json ./scripts/hooks-config-claude-code.sh
```

To install hooks for all tools at once (Claude Code + Cursor + Antigravity):

```bash
./scripts/hooks-setup.sh
```

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
