import argparse

from src.server import LibrarianServer, load_config

_TRANSPORTS = ("stdio", "sse", "streamable-http")


def main() -> None:
    parser = argparse.ArgumentParser(description="Librarian MCP server")
    parser.add_argument(
        "--transport",
        choices=_TRANSPORTS,
        default=None,
        help=(
            "Transport to use. Overrides config/env. "
            "Use 'stdio' for local MCP clients (Claude, Cursor); "
            "'sse' or 'streamable-http' for HTTP-based clients."
        ),
    )
    args = parser.parse_args()

    config = load_config()
    server = LibrarianServer(config)
    transport = args.transport or server.config.server.transport
    server.mcp.run(transport=transport)  # type: ignore[arg-type]


main()
