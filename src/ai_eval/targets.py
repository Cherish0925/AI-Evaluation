"""Adapters for recorded outputs, one-shot command processes and installed targets."""

from __future__ import annotations

import copy
import hashlib
from importlib.metadata import entry_points
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
from typing import Any


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _parse(value: str) -> Any:
    result = json.loads(value, parse_constant=_invalid_constant)
    _json(result)
    return result


def _file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


class PredictionTarget:
    """Replay JSONL predictions. Exact trials override sample-wide predictions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        raw = self.path.read_bytes()
        self.identity = {"kind": "predictions-jsonl", "sha256": hashlib.sha256(raw).hexdigest()}
        self._predictions = {}
        for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = _parse(line)
            if not isinstance(row, dict) or not isinstance(row.get("sample_id"), str) or not row["sample_id"]:
                raise ValueError(f"prediction line {number} requires a sample_id string")
            if "output" not in row:
                raise ValueError(f"prediction line {number} requires output")
            trial = row.get("trial")
            if "trial" in row and (type(trial) is not int or trial < 0):
                raise ValueError(f"prediction line {number} trial must be a nonnegative integer")
            key = row["sample_id"], trial
            if key in self._predictions:
                raise ValueError(f"duplicate prediction for sample/trial {key}")
            self._predictions[key] = {"output": row["output"], "usage": row.get("usage", {}),
                                      "metadata": row.get("metadata", {})}

    def __call__(self, request: dict) -> dict:
        key = request["sample_id"], request["trial"]
        value = self._predictions.get(key, self._predictions.get((key[0], None)))
        if value is None:
            raise KeyError(f"no prediction for sample {key[0]!r}, trial {key[1]}")
        return copy.deepcopy(value)


class CommandTarget:
    """One JSON request on stdin and one JSON response on stdout per process."""

    def __init__(self, argv: list[str], timeout: float = 60, *, cwd: str | Path | None = None):
        if (not isinstance(argv, (list, tuple)) or not argv or not isinstance(argv[0], str)
                or not argv[0] or any(not isinstance(arg, str) for arg in argv)):
            raise ValueError("command argv must be a nonempty list of strings")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("command timeout must be finite and positive")
        self.argv, self.timeout = list(argv), float(timeout)
        self.cwd = Path.cwd().resolve() if cwd is None else Path(cwd).resolve()
        if not self.cwd.is_dir():
            raise ValueError("command cwd must be a directory")

    def identity(self) -> dict:
        """Refresh command file identities once at each run/resume boundary."""
        files = {}
        executable = shutil.which(self.argv[0])
        for index, argument in enumerate(self.argv):
            path = Path(executable) if index == 0 and executable else self.cwd / argument
            if path.is_file():
                path = path.resolve()
                files[str(index)] = {"path": str(path), "sha256": _file_digest(path)}
        return {"kind": "command-json", "argv": self.argv, "timeout": self.timeout,
                "cwd": str(self.cwd), "argument_files": files}

    def __call__(self, request: dict) -> dict:
        payload = _json(request) + "\n"
        process = subprocess.Popen(self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", cwd=self.cwd,
            start_new_session=os.name != "nt")
        try:
            stdout, stderr = process.communicate(payload, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            if os.name == "nt":
                # Kill descendants before their parent disappears from the process tree.
                try:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                process.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    for pipe in (process.stdin, process.stdout, process.stderr):
                        if pipe is not None:
                            pipe.close()
                # Windows communicate uses reader threads: closing their buffered
                # streams can block while a surviving descendant owns the pipe.
            raise TimeoutError(f"command exceeded {self.timeout:g} seconds") from exc
        if process.returncode:
            raise RuntimeError(f"command exited with code {process.returncode}: {stderr[-4000:].strip()}")
        result = _parse(stdout)
        if not isinstance(result, dict) or "output" not in result:
            raise ValueError("command response must be a JSON object containing output")
        return result


class _PluginTarget:
    def __init__(self, target, identity: dict):
        self._target = target
        self._identity = identity

    def identity(self) -> dict:
        result = copy.deepcopy(self._identity)
        inner = getattr(self._target, "identity", None)
        if inner is not None:
            inner = inner() if callable(inner) else inner
            if not isinstance(inner, dict):
                raise ValueError("target identity must be an object")
            result["target"] = inner
        return _parse(_json(result))

    def __call__(self, request):
        return self._target(request)


def resolve_plugin(name: str, config: dict):
    """Resolve one ``ai_eval.targets`` factory: ``factory(config) -> callable``."""
    if not isinstance(name, str) or not name or not isinstance(config, dict):
        raise ValueError("target plugin requires a name and configuration object")
    config = _parse(_json(config))
    matches = list(entry_points(group="ai_eval.targets", name=name))
    if len(matches) != 1:
        raise ValueError(f"expected one target plugin named {name!r}, found {len(matches)}")
    point = matches[0]
    target = point.load()(copy.deepcopy(config))
    if not callable(target):
        raise ValueError("target plugin factory must return a callable")
    identity = {"kind": "plugin", "name": name, "config": config, "factory": point.value,
                "distribution_version": point.dist.version if point.dist else None}
    return _PluginTarget(target, _parse(_json(identity)))
