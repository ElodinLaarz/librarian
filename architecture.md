# Librarian Architecture

This diagram illustrates the core components and data flow of the Librarian MCP server.

```mermaid
flowchart TD
    subgraph Agent[MCP Client / Agent]
        direction LR
        S[library.search]
        I[library.ingest]
        R[library.research]
    end

    subgraph FastMCP[MCP Interface Layer]
        Router[Tool Router & Validation]
    end

    subgraph CoreLogic[Librarian Core Services]
        SearchE[Search Engine]
        Ingestor[Ingestor]
        Verifier[Verifier]
        Researcher[Researcher sub-agent]
        WebClient[Web Search / Extractor]
    end

    subgraph Storage[Storage & Embedding Layer]
        Embed[Embedding Service]
        subgraph Repositories[Repositories]
            Mongo[MongoDB backend]
            FS[Filesystem backend]
        end
    end

    %% Agent to FastMCP
    S --> Router
    I --> Router
    R --> Router

    %% FastMCP to Services
    Router -- Search Request --> SearchE
    Router -- Ingest Request --> Ingestor
    Router -- Research Request --> Researcher

    %% Cross-service dependencies
    Ingestor <--> Verifier
    Researcher --> Ingestor : Pipes findings

    %% External APIs
    Researcher --> WebClient
    Verifier --> WebClient
    WebClient -.->|Brave/Tavily/Trafilatura| WWW((Internet))

    %% Services to Storage/Embedding
    SearchE --> Embed
    SearchE --> Repositories
    Ingestor --> Embed
    Ingestor --> Repositories
    Researcher --> Repositories : Stores jobs

    %% Repositories specifics
    Embed -.->|Ollama / SentenceTransformers| LocalModels[(Local Models)]
```
