"""Comparable measurements for interchangeable QUBO solver backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from math import isclose, isfinite
from time import perf_counter
from typing import Any

from .qubo import QUBOModel, QUBOSolution, QUBOSolver
from .qubo_io import qubo_fingerprint

QUBO_BENCHMARK_SCHEMA_VERSION = 1
FeasibilityCheck = Callable[[Sequence[int]], bool]


@dataclass(frozen=True, slots=True)
class QUBOBackendResult:
    backend: str
    solver: str
    variable_count: int
    energy: float
    energy_gap_from_best_observed: float
    is_best_observed: bool
    constraint_feasible: bool | None
    solve_seconds: float
    iterations: int
    accepted_moves: int
    sample: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QUBOBenchmarkReport:
    schema_version: int
    qubo_fingerprint: str
    variable_count: int
    linear_terms: int
    quadratic_terms: int
    best_observed_energy: float
    backends: tuple[QUBOBackendResult, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["backends"] = [backend.as_dict() for backend in self.backends]
        return payload


def compare_qubo_backends(
    model: QUBOModel,
    backends: Mapping[str, QUBOSolver],
    *,
    initial_sample: Sequence[int] | None = None,
    feasibility: FeasibilityCheck | None = None,
) -> QUBOBenchmarkReport:
    """Solve one unchanged QUBO with every backend and report comparable results."""
    if not backends:
        raise ValueError("at least one QUBO backend is required")
    if any(not isinstance(label, str) or not label for label in backends):
        raise ValueError("QUBO backend labels must be non-empty strings")
    if initial_sample is not None:
        model.validate_sample(initial_sample)

    results: list[QUBOBackendResult] = []
    for label, backend in backends.items():
        solve = getattr(backend, "solve", None)
        if not callable(solve):
            raise TypeError(f"backend {label!r} does not implement solve")
        started = perf_counter()
        solution = solve(model, initial_sample=initial_sample)
        solve_seconds = perf_counter() - started
        if not isinstance(solution, QUBOSolution):
            raise TypeError(f"backend {label!r} must return QUBOSolution")
        model.validate_sample(solution.sample)
        measured_energy = model.energy(solution.sample)
        if not isfinite(solution.energy) or not isclose(
            solution.energy,
            measured_energy,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"backend {label!r} returned energy inconsistent with its sample"
            )
        if not isinstance(solution.iterations, int) or solution.iterations < 0:
            raise ValueError(f"backend {label!r} returned invalid iterations")
        if not isinstance(solution.accepted_moves, int) or solution.accepted_moves < 0:
            raise ValueError(f"backend {label!r} returned invalid accepted_moves")
        results.append(
            QUBOBackendResult(
                backend=label,
                solver=solution.solver,
                variable_count=model.variable_count,
                energy=measured_energy,
                energy_gap_from_best_observed=0.0,
                is_best_observed=False,
                constraint_feasible=(
                    bool(feasibility(solution.sample)) if feasibility else None
                ),
                solve_seconds=solve_seconds,
                iterations=solution.iterations,
                accepted_moves=solution.accepted_moves,
                sample=tuple(solution.sample),
            )
        )

    best_energy = min(result.energy for result in results)
    compared = tuple(
        replace(
            result,
            energy_gap_from_best_observed=(
                0.0
                if isclose(result.energy, best_energy, rel_tol=1e-9, abs_tol=1e-9)
                else result.energy - best_energy
            ),
            is_best_observed=isclose(
                result.energy,
                best_energy,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ),
        )
        for result in results
    )
    return QUBOBenchmarkReport(
        schema_version=QUBO_BENCHMARK_SCHEMA_VERSION,
        qubo_fingerprint=qubo_fingerprint(model),
        variable_count=model.variable_count,
        linear_terms=len(model.linear),
        quadratic_terms=len(model.quadratic),
        best_observed_energy=best_energy,
        backends=compared,
    )
