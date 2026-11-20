"""Local execution with bounded concurrency, durable observations and exact resume."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
from datetime import datetime, timezone
import hashlib
import inspect
import json
import marshal
import math
import os
from pathlib import Path
import platform
import time
import uuid

from .contracts import canonical_json, digest, load_benchmark, load_run, system_id
from .metrics import implementation, score


class _RunLease:
    """OS-owned lease: interruption releases it without deleting a lock inode."""

    def __init__(self, directory: Path):
        self.path = directory / ".run.lock"
        self.file = None

    def __enter__(self):
        self.file = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self.file.seek(0, os.SEEK_END)
                if self.file.tell() == 0:
                    self.file.write(b"0")
                    self.file.flush()
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            self.file = None
            raise ValueError("run directory already has an active writer") from exc
        return self

    def __exit__(self, *_):
        if self.file is not None:
            try:
                if os.name == "nt":
                    import msvcrt
                    self.file.seek(0)
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            finally:
                self.file.close()
                self.file = None


def _target_identity(target) -> dict:
    identity = getattr(target, "identity", None)
    if identity is not None:
        identity = identity() if callable(identity) else identity
        if not isinstance(identity, dict):
            raise ValueError("target identity must be an object")
        return json.loads(canonical_json(identity))
    function = target if inspect.isfunction(target) or inspect.ismethod(target) else target.__call__
    code = getattr(function, "__code__", None)
    return {"kind": "python-callable", "module": getattr(function, "__module__", None),
            "qualname": getattr(function, "__qualname__", type(target).__qualname__),
            "code_sha256": hashlib.sha256(marshal.dumps(code)).hexdigest() if code else None,
            "configuration": "caller-supplied system.config; external state is not introspected"}


def _seed(protocol_seed: int, sample_id: str, trial: int) -> int:
    return int(digest({"protocol_seed": protocol_seed, "sample_id": sample_id, "trial": trial})[:16], 16) % (1 << 63)


def _error(exc: Exception) -> dict:
    return {"type": type(exc).__name__, "message": str(exc)}


def _attempt(sample: dict, trial: int, benchmark: dict, system_digest: str, target) -> dict:
    started = time.monotonic()
    names = [metric["name"] for metric in benchmark["metrics"]]
    row = {"schema_version": "1.0", "benchmark_id": benchmark["benchmark_id"], "system_id": system_digest,
           "sample_id": sample["id"], "group": sample.get("group", sample["id"]),
           "slices": sample.get("slices", {}), "weight": sample.get("weight", 1), "trial": trial,
           "seed": _seed(benchmark["protocol"]["seed"], sample["id"], trial), "status": "ok", "output": None,
           "scores": dict.fromkeys(names), "errors": {}, "usage": {}, "metadata": {}}
    request = {"sample_id": sample["id"], "input": copy.deepcopy(sample["input"]), "trial": trial,
               "seed": row["seed"], "metadata": copy.deepcopy(sample.get("metadata", {}))}
    try:
        response = target(request)
        if not isinstance(response, dict) or "output" not in response:
            raise ValueError("target response must be an object containing output")
        response = json.loads(canonical_json(response))
        usage, metadata = response.get("usage", {}), response.get("metadata", {})
        if not isinstance(usage, dict) or not isinstance(metadata, dict):
            raise ValueError("target usage and metadata must be objects")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in usage.values()):
            raise ValueError("target usage must contain finite nonnegative numeric measurements")
        row.update(output=response["output"], usage=usage, metadata=metadata)
    except Exception as exc:
        row["status"] = "target_error"
        row["errors"]["target"] = _error(exc)
    else:
        for metric in benchmark["metrics"]:
            try:
                row["scores"][metric["name"]] = score(metric, copy.deepcopy(row["output"]), copy.deepcopy(sample["target"]))
            except Exception as exc:
                row["status"] = "score_error"
                row["errors"][metric["name"]] = _error(exc)
    row["elapsed_seconds"] = time.monotonic() - started
    return row


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        output.write((canonical_json(value) + "\n").encode("utf-8"))
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    if os.name != "nt":
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _recover_tail(path: Path) -> None:
    if not path.exists():
        return
    raw = path.read_bytes()
    if not raw or raw.endswith(b"\n"):
        return
    boundary = raw.rfind(b"\n") + 1
    prefix, tail = raw[:boundary], raw[boundary:]
    # An interior corrupt record is never discarded as part of tail recovery.
    for line in prefix.splitlines():
        if line.strip():
            json.loads(line)
    try:
        json.loads(tail)
    except (json.JSONDecodeError, UnicodeDecodeError):
        with path.open("r+b") as output:
            output.truncate(boundary)
            output.flush()
            os.fsync(output.fileno())
    else:
        # A complete JSON object without its delimiter is an existing outcome.
        with path.open("ab") as output:
            output.write(b"\n")
            output.flush()
            os.fsync(output.fileno())


def run(benchmark_path: str | Path, output: str | Path, *, system: dict, target,
        workers: int = 1, resume: bool = False) -> dict:
    """Execute each requested sample/trial once per durable record.

    Failures are durable outcomes too. Resume only issues missing observations;
    it does not retry recorded failures. A target owns its external deadlines.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if type(resume) is not bool or not callable(target):
        raise ValueError("resume must be boolean and target must be callable")
    benchmark = load_benchmark(benchmark_path)
    benchmark["metrics"] = [{**metric, "implementation": implementation(metric)} for metric in benchmark["metrics"]]
    system = json.loads(canonical_json(system))
    system_digest = system_id(system)
    execution = {"target": _target_identity(target),
                 "seed_algorithm": "sha256-json-first64-mod2^63-v1", "engine_version": "0.1.0",
                 "python": platform.python_version()}
    benchmark_manifest = {"id": benchmark["benchmark_id"],
        **{key: benchmark[key] for key in ("name", "revision", "sample_sha256", "protocol", "metrics")}}
    samples = [{"id": sample["id"], "group": sample.get("group", sample["id"]),
                "slices": sample.get("slices", {}), "weight": sample.get("weight", 1)}
               for sample in benchmark["samples"]]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with _RunLease(output):
        manifest_path, results_path = output / "manifest.json", output / "results.jsonl"
        if resume:
            if not manifest_path.is_file():
                raise ValueError("resume requires an existing manifest")
            # Check immutable identity before changing even an interrupted tail.
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("benchmark") != benchmark_manifest or manifest.get("system_id") != system_digest
                    or manifest.get("execution") != execution or manifest.get("samples") != samples):
                raise ValueError("resume requires the same benchmark, system and execution contract")
            if not results_path.exists():
                results_path.touch(exist_ok=False)
            _recover_tail(results_path)
            existing = load_run(output)["results"]
        else:
            if any(path.name != ".run.lock" for path in output.iterdir()):
                raise ValueError("new run output must be empty; use resume for an existing run")
            manifest = {"schema_version": "1.0", "run_id": uuid.uuid4().hex, "benchmark": benchmark_manifest,
                "system": system, "system_id": system_digest,
                "planned_trials": len(samples) * benchmark["protocol"]["trials"], "samples": samples,
                "execution": execution, "engine": {"name": "ai-evaluation", "version": "0.1.0"},
                "runtime": {"python": platform.python_version(), "platform": platform.platform(), "workers_initial": workers},
                "created_at": datetime.now(timezone.utc).isoformat(), "metadata": benchmark.get("metadata", {})}
            _atomic_json(manifest_path, manifest)
            results_path.touch(exist_ok=False)
            existing = []
        recorded = {(row["sample_id"], row["trial"]) for row in existing}
        jobs = iter((sample, trial) for sample in benchmark["samples"]
                    for trial in range(benchmark["protocol"]["trials"])
                    if (sample["id"], trial) not in recorded)
        with results_path.open("ab") as results, ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ai-eval") as pool:
            pending = set()

            def submit_one() -> bool:
                try:
                    sample, trial = next(jobs)
                except StopIteration:
                    return False
                pending.add(pool.submit(_attempt, sample, trial, benchmark, system_digest, target))
                return True

            for _ in range(workers):
                if not submit_one():
                    break
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    pending.remove(future)
                    result = future.result()
                    results.write((canonical_json(result) + "\n").encode("utf-8"))
                    results.flush()
                    os.fsync(results.fileno())
                    submit_one()
        return load_run(output)
