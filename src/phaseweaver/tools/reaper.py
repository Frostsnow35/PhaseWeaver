"""Cross-platform process-tree reaping.

Windows notes:
- `taskkill` is denied under the local restricted token, so it is only a
  fallback. The primary mechanism is a Job Object with
  JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, which terminates the whole tree when the
  job handle is closed (no orphaned grandchildren).
- Process creation uses CREATE_NEW_PROCESS_GROUP.
POSIX uses start_new_session + killpg.
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
from typing import Any

_GRACE_FLOOR_S = 1.0

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


def popen_kwargs() -> dict[str, object]:
    """Kept for callers that build their own Popen; see spawn_flags for mypy-safe use."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def spawn_flags() -> tuple[int, bool]:
    """(creationflags, start_new_session) for subprocess.Popen."""
    if os.name == "nt":
        # Windows-only attribute; mypy runs on either platform in CI.
        return (int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)), False)
    return (0, True)


_SIGTERM = int(getattr(signal, "SIGTERM", 15))
_SIGKILL = int(getattr(signal, "SIGKILL", 9))


def _signal_group(pid: int, sig: int) -> bool:
    killpg = getattr(os, "killpg", None)
    getpgid = getattr(os, "getpgid", None)
    if killpg is None or getpgid is None:  # pragma: no cover - POSIX only
        return False
    try:
        killpg(getpgid(pid), sig)
        return True
    except ProcessLookupError:
        return False


if os.name == "nt":

    class _JobBasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JobExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JobBasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


class Job:
    """Windows Job Object that kills its whole tree when closed."""

    def __init__(self, handle: int) -> None:
        self._handle = handle

    @classmethod
    def attach(cls, proc: subprocess.Popen[bytes]) -> Job | None:
        if os.name != "nt":
            return None
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            return None
        info = _JobExtendedLimits()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            kernel32.CloseHandle(handle)
            return None
        proc_handle = int(getattr(proc, "_handle", 0) or 0)
        if not proc_handle or not kernel32.AssignProcessToJobObject(handle, proc_handle):
            kernel32.CloseHandle(handle)
            return None
        return cls(int(handle))

    def terminate(self) -> None:
        if os.name != "nt" or not self._handle:
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.TerminateJobObject(self._handle, 1)

    def close(self) -> None:
        if os.name != "nt" or not self._handle:
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle(self._handle)
        self._handle = 0


def kill_tree_sync(
    proc: subprocess.Popen[bytes] | Any, grace_ms: int = 2000, job: Job | None = None
) -> None:
    """Synchronously kill the whole tree and wait for it to disappear."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        if job is not None:
            job.terminate()
        else:  # fallback; may be denied under a restricted token
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
            )
        try:
            proc.wait(timeout=max(_GRACE_FLOOR_S, grace_ms / 1000))
        except subprocess.TimeoutExpired:
            proc.kill()
        return
    try:
        if not _signal_group(proc.pid, _SIGTERM):
            return
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_ms / 1000)
        return
    except subprocess.TimeoutExpired:
        pass
    if not _signal_group(proc.pid, _SIGKILL):
        return
    try:
        proc.wait(timeout=max(_GRACE_FLOOR_S, grace_ms / 1000))
    except subprocess.TimeoutExpired:
        return


def tree_is_gone(pid: int) -> bool:
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        return False
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return str(pid) not in (out.stdout or "")
