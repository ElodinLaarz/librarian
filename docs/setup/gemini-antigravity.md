# Gemini CLI / Antigravity — Librarian MCP Setup

Antigravity connects to the Librarian via **`stdio`** transport — no Docker or HTTP server needed. The MCP process is launched automatically by Antigravity using `uv run`.

## Option A — Local (no Docker, recommended for Antigravity)

This uses the filesystem storage backend and Ollama for embeddings. Tomes are persisted to `~/.librarian_mcp/` and are accessible to any agent on the machine.

### Prerequisites

| Requirement | Notes |
| --- | --- |
| [uv](https://docs.astral.sh/uv/) | Python package manager |
| [Ollama](https://ollama.com) | Local embedding inference |
| `nomic-embed-text` model | Pulled via `ollama pull` |

### First-time setup

```bash
# 1. Clone and install dependencies
git clone https://github.com/ElodinLaarz/librarian
cd librarian
uv sync

# 2. Pull the embedding model
ollama pull nomic-embed-text

# 3. The local config already exists at config/local-fs.yaml
#    (uses file:///~/.librarian_mcp storage + Ollama embeddings)

# 4. Register the MCP in Antigravity
#    ~/.gemini/antigravity/mcp_config.json should contain:
cat > ~/.gemini/antigravity/mcp_config.json << 'EOF'
{
  "mcpServers": {
    "librarian": {
      "command": "uv",
      "args": ["run", "python", "-m", "src"],
      "env": {
        "LIBRARIAN_CONFIG": "/path/to/librarian/config/local-fs.yaml"
      },
      "cwd": "/path/to/librarian"
    }
  }
}
EOF
```

Replace `/path/to/librarian` with the actual repo path (e.g. `/home/elodin/github/librarian`).

### Storage layout

```
~/.librarian_mcp/
├── tomes/          ← one JSON file per Tome (searchable by all agents)
└── research_jobs/  ← async research job state
```

### Verify it works

```bash
cd /path/to/librarian
LIBRARIAN_CONFIG=config/local-fs.yaml uv run python -c "
import asyncio
from src.config import LibrarianConfig
from src.server import LibrarianServer
from src.services.ingestor import IngestCallOptions

config = LibrarianConfig.from_yaml('config/local-fs.yaml')

async def smoke_test():
    server = LibrarianServer(config)
    async with server.lifespan(server.mcp):
        result = await server.ingestor.ingest('Hello from Librarian!', IngestCallOptions(skip_verify=True))
        print('status:', result.status, '| tomes:', len(result.tomes))
        hits = await server.tome_repo.search('Hello', top_k=1, min_confidence=0.0)
        print('search hit:', hits[0][0].summary if hits else 'none')

asyncio.run(smoke_test())
"
```

---

## Option B — Docker + HTTP (SSE/streamable-http)

> Use this if you want MongoDB vector search, a shared server, or remote access.

The Docker Compose **`librarian`** service sets **`LIBRARIAN_SERVER_TRANSPORT=sse`** and listens on port **8000**.

### First-time setup

```bash
chmod +x scripts/*.sh scripts/lib/common.sh
./scripts/dev-up.sh
./scripts/mcp-config-http-clients.sh
```

`mcp-config-http-clients.sh` writes:

| File | Purpose |
| --- | --- |
| **`~/.librarian/mcp-http-librarian.json`** | SSE URL (`http://localhost:8000/sse`) |
| **`~/.librarian/mcp-streamable-librarian.json`** | Alternate streamable-http URL |

### Every time you work

```bash
cd /path/to/librarian
./scripts/start-stack.sh   # start
./scripts/stop-stack.sh    # stop
```

### Point Antigravity at the HTTP server

Merge the `mcpServers` entry from `~/.librarian/mcp-http-librarian.json` into `~/.gemini/antigravity/mcp_config.json`, or use the URL directly: **`http://localhost:8000/sse`**.

Override the SSE URL:

```bash
LIBRARIAN_SSE_URL=https://your-host:8000/sse ./scripts/mcp-config-http-clients.sh
```

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Missing required config: [database.uri]` | `LIBRARIAN_CONFIG` env var is not set or points to wrong file |
| `ModuleNotFoundError: sentence_transformers` | Use `provider: ollama` in config (or `uv pip install sentence-transformers`) |
| Ollama embedding error | Ensure Ollama is running (`ollama serve`) and model is pulled (`ollama pull nomic-embed-text`) |
| All search scores `1.00` | Old tomes stored with dummy embeddings — delete `~/.librarian_mcp/tomes/*.json` and re-ingest |
| HTTP 404 on `/sse` | Try `/mcp` endpoint (streamable-http); check FastMCP version |
| Connection reset | Confirm `LIBRARIAN_SERVER_TRANSPORT=sse` (not `stdio`) for the HTTP process |
