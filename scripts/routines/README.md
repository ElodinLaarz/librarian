# routined — interval-based agentic routines

Tiny daemon that fires "every N hours, do X" against a repo using whatever
agentic CLI you have installed (Claude Code, Gemini CLI, Codex CLI, Aider,
Pi, …) running in YOLO / auto-permission mode. Each fire happens in a
fresh git clone so runs cannot stomp each other.

## Files

- [routined.py](routined.py) — the daemon. Reads YAML config, schedules
  routines, fires each into its own worktree.
- [harness.py](harness.py) — adapter that launches the chosen CLI with
  the prompt piped in and the YOLO/auto flags you configured.
- [config.example.yaml](config.example.yaml) — copy to `config.yaml` and
  edit. Defines harnesses + routines.
- [prompts/issue_to_pr.md](prompts/issue_to_pr.md) — default prompt:
  pick an open issue matching a filter, implement, open a PR, address
  review-bot comments, wait for CI, merge, clean up.
- [install.sh](install.sh) — installs cron entries (or a systemd user
  unit with `--systemd`) on Linux/macOS.
- [install.ps1](install.ps1) — registers a Windows Task Scheduler task.

## Quick start

```bash
cd scripts/routines
cp config.example.yaml config.yaml
# edit config.yaml — set repo, harness, interval, and the prompt filter

# verify it loads and lists routines
python routined.py --config config.yaml --list

# fire one tick now to smoke-test (will actually run a routine if due)
python routined.py --config config.yaml --once -v

# install as a recurring job
./install.sh                 # cron (default)
./install.sh --systemd       # systemd --user unit
./install.sh --uninstall     # remove

# windows
powershell -ExecutionPolicy Bypass -File install.ps1
```

## How a routine fires

1. Daemon ticks every `--tick` seconds (default 60).
1. For each enabled routine, if `now - last_started_at >= interval`, it
   forks a worker (Linux/macOS) or spawns a detached Python process
   (Windows) so a slow routine cannot block its siblings.
1. Worker `git clone`s the repo into `worktree_root/<routine>/<run-id>/`.
1. Renders `prompt_template` with `${repo}`, `${run_id}`, `${worktree}`,
   `${routine_name}`, plus everything under `prompt_vars`.
1. Invokes the configured harness with the prompt (stdin / arg / file)
   inside the worktree. stdout+stderr stream to
   `log_dir/<routine>/<run-id>.log`.
1. On exit (or timeout), removes the worktree and updates state at
   `state_dir/<routine>.json` (`last_started_at`, `last_finished_at`,
   `last_exit_code`, `in_flight`).

State is per-routine JSON, so `--list` and external monitors can read
it without locking.

## Defining a harness

Each entry under `harnesses:` is just `cmd` plus how the prompt is fed
in:

```yaml
harnesses:
  claude:
    cmd: ["claude", "--dangerously-skip-permissions", "--print"]
    prompt_via: stdin
  codex:
    cmd: ["codex", "exec", "--full-auto"]
    prompt_via: arg            # appended as the last argv element
  aider:
    cmd: ["aider", "--yes-always", "--auto-commits", "--no-stream"]
    prompt_via: arg
  custom:
    cmd: ["my-agent", "--auto"]
    prompt_via: file
    prompt_arg: --prompt-file  # `my-agent --auto --prompt-file <path>`
    timeout_seconds: 5400
    env: { LOG_LEVEL: info }
```

The binary must be on `PATH` when the daemon runs. If you launch from
cron, set `PATH` in the crontab or pass an absolute path in `cmd`.

## Defining a routine

```yaml
routines:
  - name: my-bot
    enabled: true
    interval: 6h        # accepts s/m/h/d, or a raw integer (seconds)
    harness: claude
    repo: git@github.com:you/repo.git
    base_branch: main
    prompt_template: prompts/issue_to_pr.md
    max_concurrent: 1   # how many runs may overlap (default 1)
    prompt_vars:
      issue_filter: 'is:open is:issue label:"good first issue" no:assignee'
      branch_prefix: routine/auto
      max_minutes: 60
      reviewer_bots: "coderabbitai, github-actions"
    env:
      GH_TOKEN: ${GH_TOKEN}
```

`prompt_vars` are dropped into the template via `string.Template` —
reference them as `$name` or `${name}` in your prompt.

## Safety notes

- Every harness here runs with permission prompts disabled. Only point
  it at repos you actually want it to merge into.
- The default prompt instructs the agent to bail rather than ship
  garbage if the issue is ambiguous, CI is flaking, or it runs out of
  time. Keep that contract when you write your own prompts.
- Worktrees live under `worktree_root` and are removed after each run.
  Logs and rendered prompts are kept under `log_dir/<routine>/` for
  debugging — rotate or prune them yourself if disk matters.
- The daemon does **not** merge anything itself. All git/PR actions
  happen inside the harness, so you can audit them via the log file.
