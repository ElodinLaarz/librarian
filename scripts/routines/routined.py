"""routined — interval-based agentic-routine daemon.

Reads a YAML config of routines and fires each one on its own schedule.
Each routine clones a target repo into a fresh worktree, renders a prompt
template, and hands it to a configured harness (claude code, gemini cli,
codex, etc.) running in YOLO / auto-permission mode. The harness is
responsible for picking an issue, opening a PR, iterating with review
bots, waiting for CI, merging, and cleaning up.

Usage:
    python routined.py --config config.yaml [--once] [--tick 30]

Flags:
    --once      Run one scheduling pass and exit (handy for cron).
    --tick N    Seconds between scheduling passes when running as a daemon.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import string
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml
from harness import Harness

log = logging.getLogger("routined")


# --------------------------------------------------------------------------- #
# Config + state
# --------------------------------------------------------------------------- #

INTERVAL_RE = re.compile(r"^\s*(\d+)\s*([smhd])\s*$", re.IGNORECASE)
INTERVAL_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_interval(value: str | int) -> int:
    if isinstance(value, int):
        return value
    m = INTERVAL_RE.match(str(value))
    if not m:
        raise ValueError(f"invalid interval {value!r} (use e.g. '30m', '2h', '1d')")
    return int(m.group(1)) * INTERVAL_UNITS[m.group(2).lower()]


@dataclass
class Routine:
    name: str
    interval_seconds: int
    harness: str
    repo: str
    prompt_template: Path
    prompt_vars: dict
    enabled: bool = True
    base_branch: str = "main"
    extra_env: dict | None = None
    max_concurrent: int = 1


@dataclass
class Config:
    state_dir: Path
    log_dir: Path
    worktree_root: Path
    harnesses: dict[str, Harness]
    routines: list[Routine]


def expand(p: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p)))).resolve()


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent

    state_dir = expand(raw.get("state_dir", "~/.routined/state"))
    log_dir = expand(raw.get("log_dir", "~/.routined/logs"))
    worktree_root = expand(raw.get("worktree_root", "~/.routined/worktrees"))
    state_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    worktree_root.mkdir(parents=True, exist_ok=True)

    harnesses = {
        name: Harness.from_config(name, h_raw)
        for name, h_raw in (raw.get("harnesses") or {}).items()
    }
    if not harnesses:
        raise ValueError("config has no 'harnesses' defined")

    routines: list[Routine] = []
    for r in raw.get("routines") or []:
        if r["harness"] not in harnesses:
            raise ValueError(f"routine {r['name']!r} references unknown harness {r['harness']!r}")
        tmpl = Path(r["prompt_template"])
        if not tmpl.is_absolute():
            tmpl = (base / tmpl).resolve()
        routines.append(
            Routine(
                name=r["name"],
                interval_seconds=parse_interval(r["interval"]),
                harness=r["harness"],
                repo=r["repo"],
                prompt_template=tmpl,
                prompt_vars=dict(r.get("prompt_vars", {})),
                enabled=r.get("enabled", True),
                base_branch=r.get("base_branch", "main"),
                extra_env=r.get("env"),
                max_concurrent=int(r.get("max_concurrent", 1)),
            )
        )
    if not routines:
        raise ValueError("config has no 'routines' defined")

    return Config(
        state_dir=state_dir,
        log_dir=log_dir,
        worktree_root=worktree_root,
        harnesses=harnesses,
        routines=routines,
    )


def state_path(state_dir: Path, name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return state_dir / f"{safe}.json"


def _lock_path(state_dir: Path, name: str) -> Path:
    return state_path(state_dir, name).with_suffix(".lock")


@contextmanager
def _file_lock(path: Path):
    """Process-safe exclusive lock on a sentinel file. POSIX uses fcntl,
    Windows uses msvcrt. Held only across a state read/modify/write cycle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if sys.platform == "win32":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                with suppress(OSError):
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                with suppress(OSError):
                    fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _read_state_unlocked(state_dir: Path, name: str) -> dict:
    p = state_path(state_dir, name)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_state_atomic(state_dir: Path, name: str, data: dict) -> None:
    p = state_path(state_dir, name)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def read_state(state_dir: Path, name: str) -> dict:
    with _file_lock(_lock_path(state_dir, name)):
        return _read_state_unlocked(state_dir, name)


def write_state(state_dir: Path, name: str, data: dict) -> None:
    with _file_lock(_lock_path(state_dir, name)):
        _write_state_atomic(state_dir, name, data)


@contextmanager
def update_state(state_dir: Path, name: str):
    """Read-modify-write the state file under an exclusive lock.

    Use this whenever you need both the current state and to persist a
    change based on it (e.g. appending or removing from `in_flight`).
    """
    with _file_lock(_lock_path(state_dir, name)):
        data = _read_state_unlocked(state_dir, name)
        yield data
        _write_state_atomic(state_dir, name, data)


# --------------------------------------------------------------------------- #
# Worktree management
# --------------------------------------------------------------------------- #


def clone_worktree(repo: str, branch: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "50", "--branch", branch, repo, str(dest)],
        check=True,
    )


def cleanup_worktree(dest: Path) -> None:
    if not dest.exists():
        return
    if sys.platform == "win32":
        subprocess.run(["cmd", "/c", "rmdir", "/S", "/Q", str(dest)], check=False)
    else:
        subprocess.run(["rm", "-rf", str(dest)], check=False)


# --------------------------------------------------------------------------- #
# Scheduling
# --------------------------------------------------------------------------- #


def now_ts() -> float:
    return time.time()


def is_due(routine: Routine, state: dict) -> bool:
    last = state.get("last_started_at", 0.0)
    return (now_ts() - last) >= routine.interval_seconds


def render_prompt(routine: Routine, run_id: str, worktree: Path) -> str:
    body = routine.prompt_template.read_text(encoding="utf-8")
    vars_ = {
        "repo": routine.repo,
        "base_branch": routine.base_branch,
        "run_id": run_id,
        "worktree": str(worktree),
        "routine_name": routine.name,
        **routine.prompt_vars,
    }
    return string.Template(body).safe_substitute(vars_)


def fire_routine(cfg: Config, routine: Routine) -> None:
    run_id = f"{int(now_ts())}-{uuid.uuid4().hex[:6]}"

    # Atomically claim a slot under max_concurrent and stamp last_started_at.
    with update_state(cfg.state_dir, routine.name) as state:
        in_flight = state.setdefault("in_flight", [])
        if len(in_flight) >= routine.max_concurrent:
            log.info(
                "[%s] at max_concurrent=%d - skipping",
                routine.name,
                routine.max_concurrent,
            )
            return
        state["last_started_at"] = now_ts()
        in_flight.append(run_id)

    worktree = cfg.worktree_root / routine.name / run_id
    log_file = cfg.log_dir / routine.name / f"{run_id}.log"
    prompt_file = cfg.log_dir / routine.name / f"{run_id}.prompt.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    log.info("[%s] firing run=%s worktree=%s", routine.name, run_id, worktree)
    try:
        clone_worktree(routine.repo, routine.base_branch, worktree)
    except subprocess.CalledProcessError as e:
        log.error("[%s] clone failed: %s", routine.name, e)
        with update_state(cfg.state_dir, routine.name) as state:
            state["last_error"] = f"clone failed: {e}"
            state["in_flight"] = [r for r in state.get("in_flight", []) if r != run_id]
            state["last_finished_at"] = now_ts()
            state["last_exit_code"] = -1
            state["last_run_id"] = run_id
        return

    prompt = render_prompt(routine, run_id, worktree)
    harness = cfg.harnesses[routine.harness]

    rc = -1
    try:
        rc = harness.run(
            prompt=prompt,
            cwd=worktree,
            log_path=log_file,
            prompt_file=prompt_file,
            extra_env=routine.extra_env or {},
        )
        log.info("[%s] run=%s exit=%d", routine.name, run_id, rc)
    except Exception as e:  # noqa: BLE001
        log.exception("[%s] run=%s crashed: %s", routine.name, run_id, e)
    finally:
        cleanup_worktree(worktree)
        with update_state(cfg.state_dir, routine.name) as state:
            state["in_flight"] = [r for r in state.get("in_flight", []) if r != run_id]
            state["last_finished_at"] = now_ts()
            state["last_exit_code"] = rc
            state["last_run_id"] = run_id


def tick(cfg: Config) -> None:
    for routine in cfg.routines:
        if not routine.enabled:
            continue
        state = read_state(cfg.state_dir, routine.name)
        if not is_due(routine, state):
            continue
        # fork-and-forget so a slow routine does not block siblings
        pid = os.fork() if hasattr(os, "fork") else None
        if pid == 0:
            try:
                fire_routine(cfg, routine)
            finally:
                os._exit(0)
        elif pid is None:
            # Windows: spawn a detached subprocess of ourselves
            spawn_self_for(cfg, routine)
        else:
            log.info("[%s] dispatched pid=%d", routine.name, pid)


def spawn_self_for(cfg: Config, routine: Routine) -> None:
    cfg_path = os.environ.get("ROUTINED_CONFIG")
    if not cfg_path:
        log.error("ROUTINED_CONFIG not set — cannot spawn worker on Windows")
        return
    argv = [
        sys.executable,
        os.path.abspath(__file__),
        "--config",
        cfg_path,
        "--run-routine",
        routine.name,
    ]
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(
            subprocess, "DETACHED_PROCESS", 0x00000008
        )
    subprocess.Popen(argv, creationflags=creationflags, close_fds=True)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def daemon_loop(cfg: Config, tick_seconds: int) -> None:
    stop = {"flag": False}

    def handle(_signum, _frame):
        stop["flag"] = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(ValueError, AttributeError):
            signal.signal(sig, handle)

    # Auto-reap forked workers on POSIX so they don't pile up as zombies.
    if hasattr(signal, "SIGCHLD"):
        with suppress(ValueError, OSError):
            signal.signal(signal.SIGCHLD, signal.SIG_IGN)

    log.info("daemon started: %d routines, tick=%ds", len(cfg.routines), tick_seconds)
    while not stop["flag"]:
        try:
            tick(cfg)
        except Exception:  # noqa: BLE001
            log.exception("tick error")
        for _ in range(tick_seconds):
            if stop["flag"]:
                break
            time.sleep(1)
    log.info("daemon stopped")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="routined")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--once", action="store_true", help="run one tick and exit")
    p.add_argument("--tick", type=int, default=60, help="seconds between scheduler passes")
    p.add_argument(
        "--run-routine",
        metavar="NAME",
        help="execute a single routine in-process (used by the spawner)",
    )
    p.add_argument("--list", action="store_true", help="list configured routines and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    os.environ["ROUTINED_CONFIG"] = str(args.config.resolve())

    cfg = load_config(args.config)

    if args.list:
        for r in cfg.routines:
            state = read_state(cfg.state_dir, r.name)
            last = state.get("last_started_at", 0)
            last_str = datetime.fromtimestamp(last, tz=UTC).isoformat() if last else "never"
            print(
                f"{r.name:30s} every={r.interval_seconds}s "
                f"harness={r.harness:8s} enabled={r.enabled} last={last_str}"
            )
        return 0

    if args.run_routine:
        match = next((r for r in cfg.routines if r.name == args.run_routine), None)
        if not match:
            log.error("unknown routine %r", args.run_routine)
            return 2
        fire_routine(cfg, match)
        return 0

    if args.once:
        tick(cfg)
        return 0

    daemon_loop(cfg, args.tick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
