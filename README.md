# librarian

An MCP server that provides AI agents with a persistent, searchable, and self-expanding knowledge base. Rather than storing raw text, the Librarian acts as an active curator: it verifies incoming information against web sources, chunks it into concise "tomes," and can autonomously conduct web research when the knowledge base is insufficient.

## Tools

| Tool               | Description                                                                                               |
| ------------------ | --------------------------------------------------------------------------------------------------------- |
| `library.search`   | Semantic vector search over stored tomes, with optional category and confidence filtering                 |
| `library.ingest`   | Fact-checks agent-provided information via web search, chunks it, and stores verified tomes               |
| `library.research` | Dispatches a Researcher sub-agent to search the web, synthesise findings, and ingest results as new tomes |

### Usage pattern

`library.search` should be the agent's first call when it needs factual information. If zero results are returned with confidence > 0.5, call `library.research` to populate the library on that topic. Call `library.ingest` whenever the agent learns something new that should be persisted.

`library.research` supports an `async: true` mode that returns a `job_id` immediately — poll with the same tool to check status while the library remains searchable.

## Data Model

The core unit is a **Tome** — a compact, single-topic document (100–400 words) containing a title, content, one-to-two sentence summary, category, tags, source provenance, a `confidence` score (0.0–1.0), and a dense embedding vector. Tomes are intentionally small to keep retrieval precise.

A second collection, **ResearchJob**, tracks the state and output of each `library.research` dispatch (`pending` → `running` → `completed` / `failed`).

Data models are defined with **Pydantic**.

## Verification

Before storing, the Verifier extracts key factual claims and cross-references them against web search results:

- **confidence > 0.7** — stored as-is
- **confidence 0.3–0.7** — stored with a low-confidence flag
- **confidence \< 0.3** — rejected by default

Set `skip_verify: true` on ingest to bypass (useful for agent-generated notes or fictional content). When no search API key is configured, verification is skipped and tomes are assigned a synthetic confidence of `0.6`.

## Stack

| Component           | Technology                                                                     |
| ------------------- | ------------------------------------------------------------------------------ |
| MCP Framework       | FastMCP (Python)                                                               |
| Database            | MongoDB 7.x via `motor` (async)                                                |
| Vector Search       | Atlas Vector Search (cosine similarity)                                        |
| Embedding (default) | `nomic-embed-text` via Ollama (768 dims, ~30ms on CPU)                         |
| Embedding (alt)     | `sentence-transformers/all-MiniLM-L6-v2` (384 dims, ~15ms, no Ollama required) |
| Chunking            | LangChain `RecursiveTextSplitter`                                              |
| Claim extraction    | Instructor + local Ollama LLM                                                  |
| Web search          | Brave Search API / Tavily / Serper (configurable)                              |
| HTML extraction     | `trafilatura`                                                                  |
| Config              | Pydantic Settings + `librarian.config.yaml`                                    |

## Quick Start

```yaml
# docker-compose.yml
version: '3.8'
services:
  mongo:
    image: mongodb/mongodb-atlas-local:latest
    ports: ['27017:27017']
    volumes: ['mongo_data:/data/db']

  ollama:
    image: ollama/ollama:latest
    ports: ['11434:11434']
    volumes: ['ollama_data:/root/.ollama']
    command: serve

  librarian:
    build: .
    ports: ['8000:8000']
    environment:
      MONGODB_URI: mongodb://mongo:27017
      OLLAMA_HOST: http://ollama:11434
      SEARCH_API_KEY: ${SEARCH_API_KEY}
    depends_on: [mongo, ollama]

volumes:
  mongo_data:
  ollama_data:
```

Pull the embedding model once after starting:

```bash
docker compose exec ollama ollama pull nomic-embed-text
```

Then point any MCP client at `http://localhost:8000`.
