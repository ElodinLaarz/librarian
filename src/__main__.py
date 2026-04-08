import argparse

from src.config import _load_dotenv
from src.server import _server, mcp

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

    _load_dotenv()
    transport = args.transport or _server.config.server.transport
    mcp.run(transport=transport)  # type: ignore[arg-type]


main()
