import asyncio
import sys

from phaseweaver.tools.base import ToolSpec, ToolStatus
from phaseweaver.tools.python_exec import PythonExecutor
from phaseweaver.tools.shell_exec import ShellExecutor


def test_python_exec_returns_stdout() -> None:
    async def scenario() -> None:
        result = await PythonExecutor().run({"code": "print('hi')"}, ToolSpec("p", "python"))
        assert result.status is ToolStatus.OK
        assert "hi" in result.stdout

    asyncio.run(scenario())


def test_python_error_reported() -> None:
    async def scenario() -> None:
        result = await PythonExecutor().run(
            {"code": "raise ValueError('boom')"}, ToolSpec("p", "python")
        )
        assert result.status is ToolStatus.ERROR
        assert "ValueError" in result.stderr

    asyncio.run(scenario())


def test_timeout_kills_process() -> None:
    async def scenario() -> None:
        spec = ToolSpec("s", "shell", timeout_s=0.7, grace_ms=500)
        cmd = f'{sys.executable} -c "import time; time.sleep(30)"'
        result = await ShellExecutor().run({"command": cmd}, spec)
        assert result.status is ToolStatus.TIMEOUT
        assert result.exit_code is not None

    asyncio.run(scenario())
