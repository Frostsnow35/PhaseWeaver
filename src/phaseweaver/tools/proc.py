"""Process runner.

Two local constraints drive this module:

1. The sandbox forbids pipe creation, and on Windows `asyncio`'s subprocess
   transport always creates pipes internally. So we use `subprocess.Popen`
   directly with stdio redirected to files, and wait on a worker thread.
2. `tempfile.mkdtemp` creates directories with mode 0o700, which the local
   security policy turns into an unwritable directory. We create our own
   working directory with default permissions instead.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from phaseweaver.tools.base import ToolStatus
from phaseweaver.tools.reaper import Job, kill_tree_sync, spawn_flags

MAX_OUTPUT = 1 << 20


@dataclass
class ProcOutcome:
    status: ToolStatus
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    truncated: bool = False


def _work_root() -> Path:
    root = Path(os.environ.get("PW_TOOL_TMPDIR", ".phaseweaver-tmp"))
    root.mkdir(parents=True, exist_ok=True)
    return root


async def run_process(
    argv: list[str] | str,
    *,
    shell: bool = False,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    stdin_text: str | None = None,
    timeout_s: float = 30.0,
    grace_ms: int = 2000,
) -> ProcOutcome:
    workdir = _work_root() / f"pw-proc-{uuid.uuid4().hex[:12]}"
    workdir.mkdir(parents=True, exist_ok=True)
    job: Job | None = None
    try:
        stdin_path = workdir / "stdin.txt"
        stdout_path = workdir / "stdout.txt"
        stderr_path = workdir / "stderr.txt"
        stdin_path.write_text(stdin_text or "", encoding="utf-8", newline="")
        status = ToolStatus.OK
        creationflags, start_new_session = spawn_flags()
        with (
            stdin_path.open(encoding="utf-8") as fin,
            stdout_path.open("w", encoding="utf-8", errors="replace") as fout,
            stderr_path.open("w", encoding="utf-8", errors="replace") as ferr,
        ):
            proc = subprocess.Popen(  # noqa: S603
                argv,
                shell=shell,
                stdin=fin,
                stdout=fout,
                stderr=ferr,
                cwd=cwd,
                env=env,
                creationflags=creationflags,
                start_new_session=start_new_session,
            )
            job = Job.attach(proc)
            try:
                await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=timeout_s)
                status = ToolStatus.OK if proc.returncode == 0 else ToolStatus.ERROR
            except TimeoutError:
                await asyncio.to_thread(kill_tree_sync, proc, grace_ms, job)
                status = ToolStatus.TIMEOUT
            except asyncio.CancelledError:
                await asyncio.to_thread(kill_tree_sync, proc, grace_ms, job)
                raise
        stdout = _read(stdout_path)
        stderr = _read(stderr_path)
        if status is ToolStatus.TIMEOUT:
            stderr = (stderr + "\ntimed out").strip()
        return ProcOutcome(
            status=status,
            stdout=stdout,
            stderr=stderr,
            exit_code=proc.returncode,
            truncated=len(stdout) >= MAX_OUTPUT,
        )
    finally:
        if job is not None:
            job.close()  # KILL_ON_JOB_CLOSE: no grandchildren survive
        shutil.rmtree(workdir, ignore_errors=True)


def _read(path: Path) -> str:
    with path.open(encoding="utf-8", errors="replace") as fh:
        return fh.read(MAX_OUTPUT)
