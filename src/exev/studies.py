"""Reproducible, resumable experiment sweeps and aggregate summaries."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from itertools import product
from math import isfinite
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .disasters import DISASTER_PROFILES
from .experiments import MODEL_VERSION, ExperimentConfig, run_scenario
from .routing import ROUTER_NAMES

STUDY_SCHEMA_VERSION = 1
StatusCallback = Callable[[str], None]

SUMMARY_GROUP_FIELDS = (
    "algorithm",
    "width",
    "height",
    "agent_count",
    "disaster_profile",
    "hazard_weight",
    "qubo_batch_size",
    "qubo_routes_per_shelter",
    "qubo_congestion_weight",
    "qubo_sweeps",
    "qubo_restarts",
)
SUMMARY_METRICS = (
    "evacuated",
    "stranded",
    "unfinished",
    "evacuation_rate",
    "elapsed_ticks",
    "mean_evacuation_time",
    "max_evacuation_time",
    "p50_evacuation_time",
    "p90_evacuation_time",
    "p95_evacuation_time",
    "total_waiting_time",
    "mean_waiting_time",
    "p95_waiting_time",
    "total_distance_traveled",
    "mean_distance_traveled",
    "total_hazard_exposure",
    "mean_hazard_exposure",
    "p95_hazard_exposure",
    "max_hazard_exposure",
    "completion_time_gini",
    "hazard_exposure_gini",
    "slow_agent_mean_evacuation_time",
    "fast_agent_mean_evacuation_time",
    "speed_group_evacuation_gap",
    "peak_congestion",
    "total_edge_occupancy_ticks",
    "full_edge_ticks",
    "mean_edge_utilization",
    "preparation_seconds",
    "routing_seconds",
    "wall_seconds",
)


def _require_unique(name: str, values: tuple[Any, ...]) -> None:
    if not values:
        raise ValueError(f"{name} cannot be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique values")


@dataclass(frozen=True, slots=True)
class StudyDefinition:
    """Cartesian experiment definition with algorithm-aware parameter sweeps."""

    map_sizes: tuple[tuple[int, int], ...] = ((8, 8),)
    agent_counts: tuple[int, ...] = (100,)
    seeds: tuple[int, ...] = (7, 8, 9)
    disaster_profiles: tuple[str, ...] = ("none", "fire")
    algorithms: tuple[str, ...] = ROUTER_NAMES
    hazard_weights: tuple[float, ...] = (1.0,)
    qubo_batch_sizes: tuple[int, ...] = (8,)
    qubo_routes_per_shelter: tuple[int, ...] = (3,)
    qubo_congestion_weights: tuple[float, ...] = (1.0,)
    qubo_sweeps: tuple[int, ...] = (100,)
    qubo_restarts: tuple[int, ...] = (3,)
    max_ticks: int = 10_000
    reroute_wait_threshold: int = 10

    def __post_init__(self) -> None:
        for name, values in (
            ("map_sizes", self.map_sizes),
            ("agent_counts", self.agent_counts),
            ("seeds", self.seeds),
            ("disaster_profiles", self.disaster_profiles),
            ("algorithms", self.algorithms),
            ("hazard_weights", self.hazard_weights),
            ("qubo_batch_sizes", self.qubo_batch_sizes),
            ("qubo_routes_per_shelter", self.qubo_routes_per_shelter),
            ("qubo_congestion_weights", self.qubo_congestion_weights),
            ("qubo_sweeps", self.qubo_sweeps),
            ("qubo_restarts", self.qubo_restarts),
        ):
            _require_unique(name, values)
        if any(
            not isinstance(size, tuple)
            or len(size) != 2
            or any(not isinstance(value, int) or value < 2 for value in size)
            for size in self.map_sizes
        ):
            raise ValueError("map_sizes must contain (width, height) integer pairs of at least 2")
        if any(not isinstance(value, int) or value < 0 for value in self.agent_counts):
            raise ValueError("agent_counts must contain non-negative integers")
        if any(not isinstance(value, int) for value in self.seeds):
            raise ValueError("seeds must contain integers")
        unknown_disasters = set(self.disaster_profiles) - set(DISASTER_PROFILES)
        if unknown_disasters:
            raise ValueError(
                f"unknown disaster profiles: {', '.join(sorted(unknown_disasters))}"
            )
        unknown_algorithms = set(self.algorithms) - set(ROUTER_NAMES)
        if unknown_algorithms:
            raise ValueError(
                f"unknown routing algorithms: {', '.join(sorted(unknown_algorithms))}"
            )
        if any(not isfinite(value) or value < 0 for value in self.hazard_weights):
            raise ValueError("hazard_weights must be finite and non-negative")
        if any(
            not isfinite(value) or value < 0
            for value in self.qubo_congestion_weights
        ):
            raise ValueError("qubo_congestion_weights must be finite and non-negative")
        for name, values in (
            ("qubo_batch_sizes", self.qubo_batch_sizes),
            ("qubo_routes_per_shelter", self.qubo_routes_per_shelter),
            ("qubo_sweeps", self.qubo_sweeps),
            ("qubo_restarts", self.qubo_restarts),
        ):
            if any(not isinstance(value, int) or value < 1 for value in values):
                raise ValueError(f"{name} must contain positive integers")
        if not isinstance(self.max_ticks, int) or self.max_ticks < 1:
            raise ValueError("max_ticks must be a positive integer")
        if (
            not isinstance(self.reroute_wait_threshold, int)
            or self.reroute_wait_threshold < 1
        ):
            raise ValueError("reroute_wait_threshold must be a positive integer")


@dataclass(frozen=True, slots=True)
class StudyRun:
    run_id: str
    algorithm: str
    config: ExperimentConfig


@dataclass(frozen=True, slots=True)
class StudyOutcome:
    planned_runs: int
    completed_runs: int
    skipped_runs: int
    raw_results: Path
    summary_results: Path
    manifest: Path


def _run_id(algorithm: str, config: ExperimentConfig) -> str:
    identity = {
        "model_version": MODEL_VERSION,
        "algorithm": algorithm,
        "configuration": asdict(config),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def plan_study(definition: StudyDefinition) -> tuple[StudyRun, ...]:
    """Expand a study without duplicating algorithms for irrelevant QUBO knobs."""
    runs: list[StudyRun] = []
    defaults = ExperimentConfig()
    for (width, height), agent_count, seed, disaster in product(
        definition.map_sizes,
        definition.agent_counts,
        definition.seeds,
        definition.disaster_profiles,
    ):
        for algorithm in definition.algorithms:
            if algorithm == "qubo-sa":
                parameter_sets: Iterable[tuple[float, int, int, float, int, int]] = product(
                    definition.hazard_weights,
                    definition.qubo_batch_sizes,
                    definition.qubo_routes_per_shelter,
                    definition.qubo_congestion_weights,
                    definition.qubo_sweeps,
                    definition.qubo_restarts,
                )
            elif algorithm == "hazard-aware":
                parameter_sets = (
                    (
                        weight,
                        defaults.qubo_batch_size,
                        defaults.qubo_routes_per_shelter,
                        defaults.qubo_congestion_weight,
                        defaults.qubo_sweeps,
                        defaults.qubo_restarts,
                    )
                    for weight in definition.hazard_weights
                )
            else:
                parameter_sets = (
                    (
                        defaults.hazard_weight,
                        defaults.qubo_batch_size,
                        defaults.qubo_routes_per_shelter,
                        defaults.qubo_congestion_weight,
                        defaults.qubo_sweeps,
                        defaults.qubo_restarts,
                    ),
                )

            for (
                hazard_weight,
                batch_size,
                routes_per_shelter,
                congestion_weight,
                sweeps,
                restarts,
            ) in parameter_sets:
                config = ExperimentConfig(
                    width=width,
                    height=height,
                    agent_count=agent_count,
                    seed=seed,
                    max_ticks=definition.max_ticks,
                    reroute_wait_threshold=definition.reroute_wait_threshold,
                    disaster_profile=disaster,
                    hazard_weight=hazard_weight,
                    qubo_batch_size=batch_size,
                    qubo_routes_per_shelter=routes_per_shelter,
                    qubo_congestion_weight=congestion_weight,
                    qubo_sweeps=sweeps,
                    qubo_restarts=restarts,
                )
                runs.append(StudyRun(_run_id(algorithm, config), algorithm, config))
    if len({run.run_id for run in runs}) != len(runs):
        raise RuntimeError("study expansion produced duplicate run IDs")
    return tuple(runs)


def _source_control_metadata() -> dict[str, Any]:
    directory = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=directory,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "", "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def _artifact_paths(raw_results: Path) -> tuple[Path, Path]:
    if raw_results.suffix.lower() != ".csv":
        raise ValueError("study output must use a .csv extension")
    summary = raw_results.with_name(f"{raw_results.stem}.summary.csv")
    manifest = raw_results.with_name(f"{raw_results.stem}.manifest.json")
    return summary, manifest


def _read_existing(path: Path) -> tuple[list[dict[str, Any]], list[str] | None]:
    if not path.exists():
        return [], None
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames is None:
        return [], None
    if "run_id" not in fieldnames:
        raise ValueError("existing study CSV does not contain run_id")
    run_ids = [row["run_id"] for row in rows]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("existing study CSV contains duplicate run IDs")
    return rows, fieldnames


def _number(record: dict[str, Any], field: str) -> float | None:
    value = record.get(field)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-numeric {field} value in study results: {value!r}") from exc


def summarize_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = tuple(str(record.get(field, "")) for field in SUMMARY_GROUP_FIELDS)
        groups[key].append(record)

    summaries: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        summary: dict[str, Any] = dict(zip(SUMMARY_GROUP_FIELDS, key))
        summary["replicates"] = len(group)
        summary["complete_evacuations"] = sum(
            _number(record, "evacuation_rate") == 1.0 for record in group
        )
        for metric in SUMMARY_METRICS:
            values = [
                value
                for record in group
                if (value := _number(record, metric)) is not None
            ]
            if not values:
                summary[f"{metric}_mean"] = None
                summary[f"{metric}_std"] = None
                summary[f"{metric}_min"] = None
                summary[f"{metric}_max"] = None
                continue
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
            summary[f"{metric}_min"] = min(values)
            summary[f"{metric}_max"] = max(values)
        summaries.append(summary)
    return summaries


def _write_csv_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("cannot write an empty summary")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def run_study(
    definition: StudyDefinition,
    raw_results: Path,
    *,
    resume: bool = True,
    status: StatusCallback | None = None,
    progress_interval: int = 50,
) -> StudyOutcome:
    """Run missing experiments, flushing each row so interrupted studies resume."""
    if not isinstance(progress_interval, int) or progress_interval < 1:
        raise ValueError("progress_interval must be a positive integer")
    raw_results = raw_results.resolve()
    summary_path, manifest_path = _artifact_paths(raw_results)
    if raw_results.exists() and not resume:
        raise FileExistsError(f"study output already exists: {raw_results}")

    plan = plan_study(definition)
    planned_ids = {run.run_id for run in plan}
    records, fieldnames = _read_existing(raw_results)
    existing_ids = {str(record["run_id"]) for record in records}
    unexpected = existing_ids - planned_ids
    if unexpected:
        raise ValueError(
            "existing study CSV contains runs outside the current definition"
        )
    missing = [run for run in plan if run.run_id not in existing_ids]
    source_control = _source_control_metadata()
    if status:
        status(
            f"Study plan: {len(plan)} runs; {len(existing_ids)} already complete; "
            f"{len(missing)} remaining."
        )

    raw_results.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if fieldnames is not None else "w"
    with raw_results.open(mode, newline="", encoding="utf-8") as stream:
        writer: csv.DictWriter | None = None
        if fieldnames is not None:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
        for index, run in enumerate(missing, start=1):
            if status:
                status(
                    f"Study run {index}/{len(missing)}: {run.run_id} "
                    f"{run.algorithm} {run.config.width}x{run.config.height} "
                    f"agents={run.config.agent_count} "
                    f"disaster={run.config.disaster_profile} seed={run.config.seed}"
                )
            result = run_scenario(
                run.config,
                run.algorithm,
                status=None,
                progress_interval=progress_interval,
            )
            record = {
                "study_schema_version": STUDY_SCHEMA_VERSION,
                "run_id": run.run_id,
                **source_control,
                **result,
            }
            if writer is None:
                fieldnames = list(record)
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
            elif set(record) != set(fieldnames or ()):
                raise ValueError(
                    "existing study CSV schema does not match current result schema"
                )
            writer.writerow(record)
            stream.flush()
            records.append(record)

    summaries = summarize_records(records)
    _write_csv_atomic(summary_path, summaries)
    manifest = {
        "study_schema_version": STUDY_SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "source_control": source_control,
        "definition": asdict(definition),
        "planned_runs": len(plan),
        "completed_runs": len(records),
        "skipped_existing_runs": len(existing_ids),
        "raw_results": str(raw_results),
        "summary_results": str(summary_path),
    }
    _write_json_atomic(manifest_path, manifest)
    if status:
        status(
            f"Study complete: {len(records)}/{len(plan)} runs. "
            f"Summary saved to {summary_path}."
        )
    return StudyOutcome(
        planned_runs=len(plan),
        completed_runs=len(records),
        skipped_runs=len(existing_ids),
        raw_results=raw_results,
        summary_results=summary_path,
        manifest=manifest_path,
    )
