"""A small stdio tool surface over the installed Python implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal


def create_server(root: str | Path):
    """Create the optional MCP service with paths relative to one workspace."""
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.types import ToolAnnotations
    except ImportError as exc:
        raise ImportError('Install MCP support with python -m pip install ".[mcp]"') from exc

    from .analysis import compare, pass_at_k, summarize
    from .cli import _write_json, benchmark_info, bounded_analysis, capabilities, run_info, run_target, samples_page
    from .importers import import_gym, import_inspect
    from .report import write_report

    workspace = Path(root).resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("MCP root must be a directory")

    def path(value: str) -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            raise ValueError("Use a path relative to the evaluation workspace")
        resolved = (workspace / candidate).resolve()
        if not resolved.is_relative_to(workspace):
            raise ValueError("Path must resolve within the evaluation workspace")
        return resolved

    def analysis_result(value: dict, output: str | None) -> dict:
        result = bounded_analysis(value)
        if output is not None:
            destination = path(output)
            _write_json(destination, value)
            result["analysis_path"] = str(destination)
        return result

    read = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
    write = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
    execute = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)
    server = FastMCP("AI Evaluation", log_level="WARNING", instructions=(
        "Inspect capabilities and benchmark contracts before running an evaluation. "
        "Paths are relative to the configured evaluation workspace. Run outputs are new directories; "
        "resume preserves recorded outcomes. System name, revision and configuration identify the experiment. "
        "Summaries and comparisons omit sample arrays; provide output to persist the full analysis, "
        "or use paginated samples and HTML reports. Compare matching benchmarks and inspect coverage."
    ))

    @server.tool(annotations=read)
    def eval_capabilities() -> dict:
        """Discover operations, native scorers, import formats and installed entry points."""
        return capabilities() | {"workspace": str(workspace), "sample_page_limit": 100,
                                 "analysis_detail_limit": 20}

    @server.tool(annotations=read)
    def eval_benchmark(benchmark: str) -> dict:
        """Inspect benchmark identity, protocol, metrics, groups and slice fields."""
        return benchmark_info(path(benchmark))

    @server.tool(annotations=execute)
    def eval_run(benchmark: str, output: str, system_name: str, system_revision: str,
                 predictions: str | None = None, command: list[str] | None = None,
                 target_plugin: str | None = None, target_config: dict | None = None,
                 system_config: dict | None = None, workers: int = 1,
                 resume: bool = False, timeout: float = 60.0) -> dict:
        """Run predictions, explicit command argv or an installed target plugin; choose one."""
        return run_target(path(benchmark), path(output), system_name=system_name,
                          system_revision=system_revision, system_config=system_config,
                          predictions=path(predictions) if predictions is not None else None,
                          command=command, target_plugin=target_plugin, target_config=target_config,
                          workers=workers, resume=resume, timeout=timeout)

    @server.tool(annotations=write)
    def eval_summarize(run: str, metric: str, confidence: float = 0.95,
                       resamples: int = 2000, seed: int = 0, slice_by: str | None = None,
                       output: str | None = None) -> dict:
        """Estimate sample means and group-bootstrap intervals; optionally save full JSON."""
        value = summarize(path(run), metric, confidence=confidence, resamples=resamples,
                          seed=seed, slice_by=slice_by)
        return analysis_result(value, output)

    @server.tool(annotations=write)
    def eval_compare(left: str, right: str, metric: str, confidence: float = 0.95,
                     resamples: int = 2000, seed: int = 0, missing: Literal["error", "complete_cases"] = "error",
                     slice_by: str | None = None, output: str | None = None) -> dict:
        """Estimate paired right-minus-left differences with explicit missing-data policy."""
        value = compare(path(left), path(right), metric, confidence=confidence,
                        resamples=resamples, seed=seed, missing=missing, slice_by=slice_by)
        return analysis_result(value, output)

    @server.tool(annotations=write)
    def eval_import(source: str, output: str, source_format: Literal["gym", "inspect"],
                    scorer: str | None = None, value_key: str | None = None,
                    direction: Literal["maximize", "minimize"] = "maximize",
                    system_name: str | None = None, system_revision: str | None = None,
                    benchmark_name: str | None = None, revision: str = "1") -> dict:
        """Import Gym export-v1 or Inspect JSON sample observations into a native run."""
        source_path, destination = path(source), path(output)
        if source_format == "gym":
            if not system_name or not system_revision or not benchmark_name:
                raise ValueError("Gym import requires system_name, system_revision and benchmark_name")
            import_gym(source_path, destination, system={"name": system_name, "revision": system_revision},
                       benchmark_name=benchmark_name, revision=revision)
        elif source_format == "inspect":
            if not scorer:
                raise ValueError("Inspect import requires an explicit scorer")
            import_inspect(source_path, destination, scorer=scorer, value_key=value_key, direction=direction)
        else:
            raise ValueError("source_format must be gym or inspect")
        return run_info(destination)

    @server.tool(annotations=read)
    def eval_samples(run: str, offset: int = 0, limit: int = 20,
                     status: Literal["ok", "target_error", "score_error"] | None = None,
                     sample_id: str | None = None, slice_key: str | None = None,
                     slice_value: str | int | float | bool | None = None,
                     include_output: bool = False) -> dict:
        """Read up to 100 trials with filters; optional output previews stop at 2000 characters."""
        return samples_page(path(run), offset=offset, limit=limit, status=status,
                            sample_id=sample_id, slice_key=slice_key, slice_value=slice_value,
                            include_output=include_output)

    @server.tool(annotations=write)
    def eval_report(run: str, metric: str, output: str, against: str | None = None,
                    confidence: float = 0.95, resamples: int = 2000, seed: int = 0,
                    missing: Literal["error", "complete_cases"] = "error",
                    slice_by: str | None = None, title: str | None = None) -> dict:
        """Write portable interactive HTML containing the complete summary or paired analysis."""
        options = {"confidence": confidence, "resamples": resamples, "seed": seed, "slice_by": slice_by}
        value = (compare(path(run), path(against), metric, missing=missing, **options) if against is not None
                 else summarize(path(run), metric, **options))
        return write_report(value, path(output), title=title)

    @server.tool(annotations=write)
    def eval_pass_at_k(run: str, metric: str, k: int, output: str | None = None) -> dict:
        """Estimate success within k attempts from complete binary outcomes; optionally save JSON."""
        return analysis_result(pass_at_k(path(run), metric, k), output)

    @server.tool(annotations=write)
    def eval_calibrate_judge(source: str, labels: list[str] | None = None, group_key: str = "group",
                             confidence: float = 0.95, resamples: int = 2000, seed: int = 0,
                             output: str | None = None) -> dict:
        """Compare categorical judge labels to reference judgments; references are not assumed truth."""
        from .judgments import calibrate
        value = calibrate(path(source), labels=labels, group_key=group_key,
                          confidence=confidence, resamples=resamples, seed=seed)
        return analysis_result(value, output)

    return server
