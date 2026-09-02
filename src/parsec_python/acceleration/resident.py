"""Local resident worker for amortizing Python and CUDA process startup.

The worker accepts only command-line argument lists from the same local user,
then calls the ordinary accelerated CLI sequentially in a long-lived process.
Every request reparses its input and constructs a fresh physical system and
SCF state.  Only process/runtime state is retained: imported extension
modules, the CUDA context, compiled kernels, and allocator pools.

This module intentionally imports only the Python standard library.  The
folder-local ``main.py`` can therefore submit a request without importing
NumPy, SciPy, CuPy, or the accelerated package in the short-lived client.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import gc
import io
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import traceback
from typing import Callable, Sequence

from multiprocessing.connection import Client, Listener


_PROTOCOL = 1
# ``resident.py`` lives in ``src/parsec_python/acceleration``.  Keep runtime
# discovery/cache files at the repository root, not inside ``src``.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_STATE_DIRECTORY = _REPOSITORY_ROOT / ".parsec_cache"
_DISCOVERY_PATH = _STATE_DIRECTORY / "accelerated-resident.json"
_SERVER_LOG_PATH = _STATE_DIRECTORY / "accelerated-resident.log"


def _worker_environment() -> dict[str, str]:
    """Return the inherited environment with a tiny-LAPACK thread default.

    The GPU eigensolvers deliberately send only projected matrices of order
    at most 64 to host LAPACK.  A many-thread OpenBLAS team costs more to wake
    than these eigensystems take to solve and can contend with CUDA launch
    orchestration.  Respect an explicit standard OpenBLAS override; otherwise
    keep those tiny host solves serial.  This does not cap the independent
    OpenMP teams used by the native real-space kernels.
    """

    environment = os.environ.copy()
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault(
        "PARSEC_CUPY_TEMP_DIR",
        str(_STATE_DIRECTORY / "cupy-temp"),
    )
    return environment


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_discovery() -> dict[str, object]:
    payload = json.loads(_DISCOVERY_PATH.read_text(encoding="utf-8"))
    if int(payload.get("protocol", -1)) != _PROTOCOL:
        raise RuntimeError("resident worker protocol does not match this code")
    address = payload.get("address")
    token = payload.get("auth_token")
    if (
        not isinstance(address, list)
        or len(address) != 2
        or not isinstance(address[0], str)
        or not isinstance(address[1], int)
        or not isinstance(token, str)
    ):
        raise RuntimeError("resident worker discovery file is invalid")
    return payload


def _connect(payload: dict[str, object]):
    host, port = payload["address"]
    token = bytes.fromhex(str(payload["auth_token"]))
    return Client((host, int(port)), authkey=token)


def _request(message: dict[str, object]) -> dict[str, object]:
    discovery = _read_discovery()
    with _connect(discovery) as connection:
        connection.send(message)
        response = connection.recv()
    if not isinstance(response, dict):
        raise RuntimeError("resident worker returned an invalid response")
    return response


def is_running() -> bool:
    try:
        response = _request({"command": "ping"})
    except (OSError, EOFError, ValueError, RuntimeError, json.JSONDecodeError):
        return False
    return bool(response.get("ok"))


def start(server_script: os.PathLike[str] | str, timeout: float = 30.0) -> None:
    """Start a hidden local worker unless a compatible one already responds."""

    if is_running():
        return
    _STATE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(server_script).resolve()), "--resident-server"]
    with _SERVER_LOG_PATH.open("a", encoding="utf-8") as stream:
        options: dict[str, object] = {
            "cwd": str(_REPOSITORY_ROOT),
            "env": _worker_environment(),
            "stdin": subprocess.DEVNULL,
            "stdout": stream,
            "stderr": stream,
            "close_fds": True,
        }
        if os.name == "nt":
            options["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_NO_WINDOW
            )
        else:
            options["start_new_session"] = True
        subprocess.Popen(command, **options)

    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if is_running():
            return
        time.sleep(0.05)
    raise RuntimeError(
        "resident worker did not become ready; inspect "
        f"{_SERVER_LOG_PATH}"
    )


def submit(arguments: Sequence[str], server_script: os.PathLike[str] | str) -> int:
    """Submit one ordinary CLI invocation, auto-starting the worker."""

    start(server_script)
    response = _request(
        {
            "command": "run",
            "argv": list(arguments),
            "cwd": str(Path.cwd()),
        }
    )
    stdout = str(response.get("stdout", ""))
    stderr = str(response.get("stderr", ""))
    if stdout:
        print(stdout, end="")
    if stderr:
        print(stderr, end="", file=sys.stderr)
    return int(response.get("exit_code", 1))


def stop() -> bool:
    """Request a clean shutdown; return false if no worker responds."""

    try:
        response = _request({"command": "stop"})
    except (OSError, EOFError, ValueError, RuntimeError, json.JSONDecodeError):
        return False
    return bool(response.get("ok"))


def execute_request(
    payload: dict[str, object],
    runner: Callable[[Sequence[str]], int],
) -> dict[str, object]:
    """Run one validated request while capturing the ordinary CLI streams."""

    arguments = payload.get("argv")
    working_directory = payload.get("cwd")
    if (
        not isinstance(arguments, list)
        or not all(isinstance(value, str) for value in arguments)
        or not isinstance(working_directory, str)
    ):
        return {
            "exit_code": 2,
            "stdout": "",
            "stderr": "Invalid resident calculation request.\n",
        }
    requested_directory = Path(working_directory).resolve()
    if not requested_directory.is_dir():
        return {
            "exit_code": 2,
            "stdout": "",
            "stderr": f"Resident working directory does not exist: {requested_directory}\n",
        }

    output = io.StringIO()
    errors = io.StringIO()
    previous_directory = Path.cwd()
    try:
        os.chdir(requested_directory)
        with redirect_stdout(output), redirect_stderr(errors):
            try:
                exit_code = int(runner(arguments))
            except SystemExit as error:
                exit_code = int(error.code or 0)
            except BaseException:
                traceback.print_exc(file=errors)
                exit_code = 1
    finally:
        os.chdir(previous_directory)
        # Release per-calculation Python/CuPy objects while deliberately
        # retaining CuPy's allocator pool and compiled module caches.
        gc.collect()
    return {
        "exit_code": exit_code,
        "stdout": output.getvalue(),
        "stderr": errors.getvalue(),
    }


def serve() -> int:
    """Warm optional runtimes, publish discovery, and serve sequential jobs."""

    # Heavy imports and CUDA/native initialization happen before discovery is
    # published, so a client never mistakes a half-warmed worker for ready.
    # This process marker enables bounded immutable preparation caches.  It is
    # intentionally absent from ordinary modular/CLI calls, where every call
    # continues to construct and time a fresh reference system.
    os.environ["PARSEC_ACCELERATED_RESIDENT"] = "1"
    from parsec_python.acceleration.cli import main as cli_main
    from parsec_python.acceleration.backends.cupy import require_cupy, synchronize
    from parsec_python.acceleration.backends.native import _load_native

    try:
        require_cupy()
        synchronize()
    except Exception:
        # A CPU-only installation can still use the resident process to
        # amortize Python/SciPy/native imports and will report normal backend
        # fallbacks through the ordinary CLI.
        pass
    try:
        _load_native()
    except Exception:
        pass

    auth_token = secrets.token_bytes(32)
    listener = Listener(("127.0.0.1", 0), authkey=auth_token)
    host, port = listener.address
    _atomic_json(
        _DISCOVERY_PATH,
        {
            "protocol": _PROTOCOL,
            "pid": os.getpid(),
            "address": [str(host), int(port)],
            "auth_token": auth_token.hex(),
            "python": sys.executable,
            "started": time.time(),
        },
    )
    try:
        running = True
        while running:
            connection = listener.accept()
            try:
                payload = connection.recv()
                if not isinstance(payload, dict):
                    response = {"ok": False, "error": "invalid request"}
                elif payload.get("command") == "ping":
                    response = {"ok": True, "pid": os.getpid()}
                elif payload.get("command") == "stop":
                    response = {"ok": True, "pid": os.getpid()}
                    running = False
                elif payload.get("command") == "run":
                    response = execute_request(payload, cli_main)
                else:
                    response = {"ok": False, "error": "unknown command"}
                connection.send(response)
            except (EOFError, OSError):
                pass
            finally:
                connection.close()
    finally:
        listener.close()
        try:
            current = _read_discovery()
            if int(current.get("pid", -1)) == os.getpid():
                _DISCOVERY_PATH.unlink(missing_ok=True)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            pass
    return 0


def status_text() -> str:
    if not is_running():
        return "Accelerated resident worker is not running."
    discovery = _read_discovery()
    return (
        "Accelerated resident worker is running "
        f"(PID {discovery['pid']}, Python {discovery['python']})."
    )


__all__ = [
    "execute_request",
    "is_running",
    "serve",
    "start",
    "status_text",
    "stop",
    "submit",
]
