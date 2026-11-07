"""Reaper coverage for the paths the happy-path tests never reach (audit F5).

The reaper kills process trees; a wrong answer here means orphaned tool processes.
These tests only use PIDs we own, and never assume the kill succeeded blindly -
they assert the detection function agrees with reality.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from phaseweaver.tools.reaper import Job, kill_tree_sync, popen_kwargs, spawn_flags, tree_is_gone


def test_spawn_flags_are_a_pair() -> None:
    flags, start_new_session = spawn_flags()
    assert isinstance(flags, int)
    assert isinstance(start_new_session, bool)
    assert isinstance(popen_kwargs(), dict)


def test_tree_is_gone_for_exited_process() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    pid = proc.pid
    proc.wait(timeout=10)
    deadline = time.monotonic() + 5
    while not tree_is_gone(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert tree_is_gone(pid)


def test_kill_tree_of_a_live_process() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **popen_kwargs(),  # type: ignore[arg-type]
    )
    time.sleep(0.3)
    assert not tree_is_gone(proc.pid)
    kill_tree_sync(proc)
    proc.wait(timeout=10)
    deadline = time.monotonic() + 5
    while not tree_is_gone(proc.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert tree_is_gone(proc.pid)


def test_job_attach_and_close_on_non_posix() -> None:
    """On Windows the Job object must attach; on POSIX it is a no-op by design."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    job = Job.attach(proc)  # type: ignore[arg-type]
    if os.name == "nt":
        assert job is not None
        job.close()
    else:
        assert job is None
    proc.wait(timeout=10)
