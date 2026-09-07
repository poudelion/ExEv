"""Small, reproducible comparisons of routing under static or dynamic hazards."""

from __future__ import annotations

import platform
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from math import isfinite
from typing import Any

from .disasters import DISASTER_PROFILES, grid_disaster_schedule
from .routing import ROUTER_NAMES, create_router
from .scenarios import grid_scenario
from .simulation import EvacuationSimulation, SimulationConfig


MODEL_VERSION = "stage8-osm-v1"
StatusCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    width: int = 12
    height: int = 12
    agent_count: int = 5_000
    seed: int = 7
    max_ticks: int = 10_000
    reroute_wait_threshold: int = 10
    disaster_profile: str = "none"
    hazard_weight: float = 1.0
    qubo_batch_size: int = 8
    qubo_routes_per_shelter: int = 3
    qubo_congestion_weight: float = 1.0
    qubo_sweeps: int = 100
    qubo_restarts: int = 3

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("grid dimensions must be at least 2")
        if self.agent_count < 0:
            raise ValueError("agent_count must be nonnegative")
        if self.max_ticks < 1:
            raise ValueError("max_ticks must be at least 1")
        if self.reroute_wait_threshold < 1:
            raise ValueError("reroute_wait_threshold must be at least 1")
        if self.disaster_profile not in DISASTER_PROFILES:
            raise ValueError(f"unknown disaster profile: {self.disaster_profile}")
        if not isfinite(self.hazard_weight) or self.hazard_weight < 0:
            raise ValueError("hazard_weight must be finite and non-negative")
        if not isinstance(self.qubo_batch_size, int) or self.qubo_batch_size < 1:
            raise ValueError("qubo_batch_size must be a positive integer")
        if not isinstance(self.qubo_routes_per_shelter, int) or self.qubo_routes_per_shelter < 1:
            raise ValueError("qubo_routes_per_shelter must be a positive integer")
        if not isfinite(self.qubo_congestion_weight) or self.qubo_congestion_weight < 0:
            raise ValueError("qubo_congestion_weight must be finite and non-negative")
        if not isinstance(self.qubo_sweeps, int) or self.qubo_sweeps < 1:
            raise ValueError("qubo_sweeps must be a positive integer")
        if not isinstance(self.qubo_restarts, int) or self.qubo_restarts < 1:
            raise ValueError("qubo_restarts must be a positive integer")


def runtime_metadata() -> dict[str, str]:
    # Source version avoids stale installed metadata when using PYTHONPATH.
    from . import __version__

    return {
        "model_version": MODEL_VERSION,
        "package_version": __version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def run_scenario(
    config: ExperimentConfig,
    algorithm: str = "dijkstra",
    *,
    status: StatusCallback | None = None,
    progress_interval: int = 50,
) -> dict[str, Any]:
    """Run one algorithm on fresh state, returning a flat exportable record."""
    if progress_interval < 1:
        raise ValueError("progress_interval must be at least 1")
    router = create_router(
        algorithm,
        qubo_routes_per_shelter=config.qubo_routes_per_shelter,
        qubo_congestion_weight=config.qubo_congestion_weight,
        hazard_weight=config.hazard_weight,
        qubo_batch_size=config.qubo_batch_size,
        qubo_sweeps=config.qubo_sweeps,
        qubo_restarts=config.qubo_restarts,
        seed=config.seed,
    )
    label = f"[{algorithm} disaster={config.disaster_profile} seed={config.seed}]"
    if status:
        status(
            f"{label} Preparing {config.agent_count:,} agents on a "
            f"{config.width} x {config.height} grid..."
        )
    network, agents, shelters = grid_scenario(
        config.width, config.height, config.agent_count, config.seed
    )
    disaster_schedule = grid_disaster_schedule(
        config.width, config.height, config.agent_count, config.disaster_profile
    )
    simulation = EvacuationSimulation(
        network,
        agents,
        shelters,
        router=router,
        config=SimulationConfig(
            max_ticks=config.max_ticks,
            reroute_wait_threshold=config.reroute_wait_threshold,
        ),
        disaster_schedule=disaster_schedule,
    )

    def report(sim: EvacuationSimulation) -> None:
        snapshot = sim.result()
        assert status is not None
        status(
            f"{label} tick={sim.tick} evacuated={snapshot.evacuated:,}/"
            f"{snapshot.total_agents:,} active={snapshot.unfinished:,} "
            f"stranded={snapshot.stranded:,}"
        )

    result = simulation.run(
        progress=report if status else None,
        progress_interval=progress_interval,
    )
    if status:
        status(f"{label} Finished in {result.wall_seconds:.3f} wall-clock seconds.")
    record = {**runtime_metadata(), **asdict(config), **asdict(result)}
    record["evacuation_rate"] = result.evacuation_rate
    return record


def compare_routers(
    config: ExperimentConfig,
    seeds: Iterable[int] | None = None,
    algorithms: Iterable[str] = ROUTER_NAMES,
    *,
    status: StatusCallback | None = None,
    progress_interval: int = 50,
) -> dict[str, Any]:
    """Rebuild state for every algorithm/seed pair and retain each result."""
    selected_seeds = list(seeds) if seeds is not None else [config.seed]
    selected_algorithms = list(algorithms)
    if not selected_seeds or not selected_algorithms:
        raise ValueError("at least one seed and algorithm are required")
    if len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("comparison seeds must be unique")
    if len(set(selected_algorithms)) != len(selected_algorithms):
        raise ValueError("comparison algorithms must be unique")
    unknown = set(selected_algorithms) - set(ROUTER_NAMES)
    if unknown:
        raise ValueError(f"unknown routing algorithms: {', '.join(sorted(unknown))}")
    records = [
        run_scenario(
            replace(config, seed=seed), algorithm,
            status=status, progress_interval=progress_interval,
        )
        for seed in selected_seeds
        for algorithm in selected_algorithms
    ]
    scenario = asdict(config)
    scenario.pop("seed")
    return {
        "schema_version": 1,
        "metadata": runtime_metadata(),
        "configuration": scenario,
        "seeds": selected_seeds,
        "algorithms": selected_algorithms,
        "records": records,
    }
