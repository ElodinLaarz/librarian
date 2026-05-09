"""Harness adapters: launch an agentic CLI in YOLO mode with a prompt."""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# argv length safety margin under POSIX ARG_MAX (typically 128 KiB - 2 MiB).
# Use a conservative cap so prompt_via='arg' fails fast with a clear message
# rather than at exec() time with a confusing "Argument list too long".
_ARG_PROMPT_MAX = 96 * 1024


@dataclass
class Harness:
    name: str
    cmd: list[str]
    prompt_via: str = "stdin"  # "stdin" | "arg" | "file"
    prompt_arg: str | None = None  # used when prompt_via == "file"
    env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: int | None = None

    @classmethod
    def from_config(cls, name: str, raw: dict) -> Harness:
        cmd = raw.get("cmd")
        if not cmd or not isinstance(cmd, list):
            raise ValueError(f"harness {name!r}: 'cmd' must be a non-empty list")
        return cls(
            name=name,
            cmd=list(cmd),
            prompt_via=raw.get("prompt_via", "stdin"),
            prompt_arg=raw.get("prompt_arg"),
            env=dict(raw.get("env", {})),
            timeout_seconds=raw.get("timeout_seconds"),
        )

    def resolve_binary(self) -> str:
        binary = self.cmd[0]
        resolved = shutil.which(binary)
        if not resolved:
            raise FileNotFoundError(f"harness {self.name!r}: binary {binary!r} not on PATH")
        return resolved

    def build_invocation(self, prompt: str, prompt_file: Path) -> list[str]:
        argv = [self.resolve_binary(), *self.cmd[1:]]
        if self.prompt_via == "arg":
            if len(prompt.encode("utf-8")) > _ARG_PROMPT_MAX:
                raise ValueError(
                    f"harness {self.name!r}: prompt is {len(prompt)} chars, "
                    f"too large for prompt_via='arg'. Switch to 'stdin' or 'file'."
                )
            argv.append(prompt)
        elif self.prompt_via == "file":
            if not self.prompt_arg:
                raise ValueError(f"harness {self.name!r}: prompt_via='file' requires 'prompt_arg'")
            argv.extend([self.prompt_arg, str(prompt_file)])
        return argv

    def run(
        self,
        prompt: str,
        cwd: Path,
        log_path: Path,
        prompt_file: Path,
        extra_env: dict[str, str] | None = None,
    ) -> int:
        prompt_file.write_text(prompt, encoding="utf-8")
        argv = self.build_invocation(prompt, prompt_file)
        env = {**os.environ, **self.env, **(extra_env or {})}

        # Spawn in its own process group / job so a timeout kill cascades to
        # any sub-processes the harness itself spawned (gh, git, node, etc.).
        popen_kwargs: dict = {
            "cwd": str(cwd),
            "env": env,
            "stdin": (subprocess.PIPE if self.prompt_via == "stdin" else subprocess.DEVNULL),
        }
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab", buffering=0) as log_fh:
            log_fh.write(f"$ cd {cwd}\n".encode())
            log_fh.write(f"$ {shlex.join(argv)}\n".encode())
            log_fh.flush()
            proc = subprocess.Popen(
                argv,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                **popen_kwargs,
            )
            if self.prompt_via == "stdin":
                assert proc.stdin is not None
                proc.stdin.write(prompt.encode("utf-8"))
                proc.stdin.close()
            try:
                return proc.wait(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                _kill_tree(proc)
                log_fh.write(b"\n[routined] TIMEOUT - killed process group\n")
                return 124


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the harness process and any descendants it spawned."""
    if sys.platform == "win32":
        # CTRL_BREAK reaches the whole process group; if anything survives,
        # taskkill /T /F reaps the descendant tree.
        with _suppress_oserror():
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass
        with _suppress_oserror():
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                check=False,
                capture_output=True,
            )
    else:
        # start_new_session=True made proc.pid the process-group leader.
        with _suppress_oserror():
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass
        with _suppress_oserror():
            os.killpg(proc.pid, signal.SIGKILL)
    with _suppress_oserror():
        proc.wait(timeout=5)


class _suppress_oserror:
    def __enter__(self):  # noqa: D401
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            return False
        if isinstance(exc, (OSError, ProcessLookupError)):
            log.debug("ignoring %s during process-tree kill", exc)
            return True
        return False
