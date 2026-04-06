# Gemini CLI / Antigravity — Librarian MCP (HTTP)

These clients typically connect to a **running** MCP server over **HTTP** (often **SSE**). The Docker Compose **`librarian`** service already sets **`LIBRARIAN_SERVER_TRANSPORT=sse`** and listens on port **8000**.

## Prerequisites

- Docker and uv (for local dev) — or any host where Librarian runs with **`sse`** (or **`streamable-http`**) transport
- jq (for generating snippet files)

## First-time setup

```bash
chmod +x scripts/*.sh scripts/lib/common.sh
./scripts/dev-up.sh
./scripts/mcp-config-http-clients.sh
```

- **`dev-up.sh`** starts the stack and ensures **`librarian.config.yaml`** exists on the host (for reference; the **container** uses compose `environment` for DB/embeddings).
- **`mcp-config-http-clients.sh`** writes:

| File | Purpose |
| --- | --- |
| **`~/.librarian/mcp-http-librarian.json`** | SSE URL (default **`http://localhost:8000/sse`**) |
| **`~/.librarian/mcp-streamable-librarian.json`** | Alternate URL for **streamable-http** clients (default **`http://localhost:8000/mcp`**) |

Override the SSE base URL:

```bash
LIBRARIAN_SSE_URL=https://your-host:8000/sse ./scripts/mcp-config-http-clients.sh
```

Override streamable URL:

```bash
LIBRARIAN_STREAMABLE_HTTP_URL=https://your-host:8000/mcp ./scripts/mcp-config-http-clients.sh
```

Output directory:

```bash
LIBRARIAN_MCP_SNIPPET_DIR=/tmp/librarian-mcp ./scripts/mcp-config-http-clients.sh
```

## Every time you work

Start (or ensure) the stack is up:

```bash
cd /path/to/librarian
./scripts/start-stack.sh
```

Stop:

```bash
./scripts/stop-stack.sh
```

No need to run a separate MCP bash process for HTTP clients: the **librarian** container serves MCP over HTTP.

## Point Gemini / Antigravity at the server

Exact UI and config file paths change between products and versions. Use the generated JSON as the source of truth:

1. Open **`~/.librarian/mcp-http-librarian.json`**.
1. Merge the `mcpServers` entry into your client’s MCP configuration, **or** set the server URL to **`http://localhost:8000/sse`** (or your published host/port).

If the client fails to connect:

- Try **`streamable-http`** and the URL in **`mcp-streamable-librarian.json`**.
- Confirm the port is reachable (firewall, Docker publish, reverse proxy).
- For **remote** Antigravity, replace `localhost` with a tunnel or public hostname.

## Run HTTP MCP on the host (without Docker librarian service)

```bash
export LIBRARIAN_SERVER_TRANSPORT=sse
export LIBRARIAN_CONFIG=/path/to/librarian.config.yaml
uv run python -m src --transport sse
```

Ensure **`librarian.config.yaml`** matches your MongoDB and embedding setup.

## Troubleshooting

- **404 on `/sse` or `/mcp`**: Check FastMCP / version for the exact path; try the other snippet file.
- **Connection reset**: Confirm **`LIBRARIAN_SERVER_TRANSPORT`** is **`sse`** (or **`streamable-http`**) for the process you are hitting, not **stdio**.
- **TLS termination**: Put HTTPS on a reverse proxy and set **`LIBRARIAN_SSE_URL`** to the public **`https://.../sse`** URL when regenerating snippets.
