# Claude Code hooks

Two hooks give Claude Code a persistent memory loop backed by the librarian:

| Hook | Event | Script | What it does |
| --- | --- | --- | --- |
| **Search** | `UserPromptSubmit` | `scripts/librarian-hook.py` | Searches the library with each prompt (20+ chars) and injects the top matches as `<librarian_context>` before Claude responds |
| **Memory sync** | `PostToolUse` on `Write\|Edit` | `scripts/claude-memory-sync-hook.py` | Mirrors Claude Code auto-memory files (`.claude/projects/*/memory/*.md`) into the library as category `agent-memory`, so per-project memories are searchable from anywhere |

Together they close the loop: anything Claude deems memory-worthy is written to file memory as usual, the sync hook mirrors it into the library, and the search hook surfaces it — in any project, on the next prompt that needs it.

## Quick install (macOS / Linux / Git Bash with `jq`)

```bash
./scripts/hooks-config-claude-code.sh
```

This merges both hooks into `~/.claude/settings.json`, preserving existing hooks. Restart Claude Code (or open `/hooks` once) to pick up the change.

## Manual install (no `jq` — e.g. plain Windows)

Merge this into `~/.claude/settings.json` (adjust the two absolute paths; on Windows use forward slashes and the full path to `uv.exe`, typically `~/.local/bin/uv.exe`):

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "shell": "bash",
            "command": "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --directory '/path/to/librarian' --quiet python '/path/to/librarian/scripts/librarian-hook.py' 2>/dev/null",
            "timeout": 30,
            "statusMessage": "Searching librarian library…"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "shell": "bash",
            "command": "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --directory '/path/to/librarian' --quiet python '/path/to/librarian/scripts/claude-memory-sync-hook.py'",
            "async": true,
            "timeout": 180,
            "statusMessage": "Syncing memory to librarian…"
          }
        ]
      }
    ]
  }
}
```

Both scripts locate the repo from their own path, so no `LIBRARIAN_REPO` env var is needed; set it only to point at a different checkout. Config resolution follows `LIBRARIAN_CONFIG` → `<repo>/librarian.config.yaml` → `<repo>/config/local-fs.yaml`.

## Behavior notes

**Search hook latency.** Each invocation is a fresh process that loads the embedding model, so expect several seconds per prompt (~9 s observed with sentence-transformers on CPU). `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` skips Hugging Face hub checks for a locally cached model. The 30 s timeout fails open — a slow or broken hook never blocks the prompt.

**Confidence and retrieval.** The search hook filters at `min_confidence` 0.55. Ingests with `skip_verify` store at the unverified default (0.5) unless a `confidence` is passed — the sync hook stores at 0.75 for this reason. If you ingest first-hand notes through `library_ingest` with `skip_verify`, pass `confidence` (or raise it later via `library_update`) or the search hook will silently never surface them.

**Memory sync semantics.** Replace-by-source: on every sync, tomes previously mirrored from the same file (`source_url`) are deleted and the current content re-ingested — the file is canonical. Do not rely on similarity dedup for this; racing near-duplicate merges interleave stale and fresh fragments when edits land seconds apart. Concurrent runs for the same file serialize on a lock (under the system temp dir) and re-read the file after acquiring it, so the last writer wins. `MEMORY.md` index files are ignored, as are writes outside `memory/` directories (gated in ~0.15 s, before any heavy import).

## Testing a hook by hand

Pipe a synthesized payload into the script — from a **file**, not an inline `echo` string. Shell escaping mangles the doubled backslashes of Windows paths in JSON, which makes a perfectly good hook look broken:

```bash
cat > /tmp/payload.json <<'EOF'
{"tool_name": "Write", "tool_input": {"file_path": "C:\\Users\\me\\.claude\\projects\\proj\\memory\\fact.md"}}
EOF
uv run --directory /path/to/librarian --quiet python scripts/claude-memory-sync-hook.py < /tmp/payload.json
```

For the search hook: `echo '{"prompt":"some question here"}' | uv run ... scripts/librarian-hook.py` (safe inline — no backslashes).

## Troubleshooting

- **Hook runs but injects nothing:** config resolution failed (check `LIBRARIAN_CONFIG` and that the config file exists), or every match fell below `min_confidence` 0.55 — see the confidence note above.
- **`connection closed` to the Mongo host from a sandboxed shell:** some agent sandboxes block outbound non-HTTP ports (27017). Hooks spawned by Claude Code run unsandboxed; only manual tests inside a sandboxed shell hit this.
- **Search misses a tome ingested seconds ago:** Atlas indexes asynchronously — read-after-write can false-negative for a few seconds.
- **Python output looks truncated or crashes on Windows:** redirected stdout defaults to a legacy codepage; both hook scripts print ASCII-safe output, but if you modify them, reconfigure stdout to UTF-8 before printing tome content.
- Hook execution logs: run `claude --debug`, or check `/hooks` in the Claude Code UI.
