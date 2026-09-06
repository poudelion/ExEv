from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from .disasters import DISASTER_PROFILES
from .experiments import ExperimentConfig, compare_routers, run_scenario
from .routing import ROUTER_NAMES


def _integer_at_least(minimum: int):
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be an integer") from exc
        if number < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}")
        return number

    return parse


def _nonnegative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not 0 <= number < float("inf"):
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or compare ExEv routing under static or dynamic disasters",
        allow_abbrev=False,
    )
    parser.add_argument("--agents", type=_integer_at_least(0), default=5_000)
    parser.add_argument("--width", type=_integer_at_least(2), default=12)
    parser.add_argument("--height", type=_integer_at_least(2), default=12)
    parser.add_argument("--seed", type=int, help="scenario seed (default: 7)")
    parser.add_argument("--max-ticks", type=_integer_at_least(1), default=10_000)
    parser.add_argument("--reroute-wait", type=_integer_at_least(1), default=10)
    parser.add_argument(
        "--disaster", choices=DISASTER_PROFILES, default="none",
        help="dynamic disaster profile (default: none)",
    )
    parser.add_argument(
        "--hazard-weight", type=_nonnegative_float, default=1.0,
        help="exposure penalty for hazard-aware and QUBO routing (default: 1.0)",
    )
    parser.add_argument(
        "--qubo-batch-size", type=_integer_at_least(1), default=8,
        help="agents per QUBO subproblem (default: 8)",
    )
    parser.add_argument(
        "--qubo-routes-per-shelter", type=_integer_at_least(1), default=3,
        help="candidate paths per shelter in each QUBO (default: 3)",
    )
    parser.add_argument(
        "--qubo-congestion-weight", type=_nonnegative_float, default=1.0,
        help="shared-road penalty in the QUBO objective (default: 1.0)",
    )
    parser.add_argument(
        "--qubo-sweeps", type=_integer_at_least(1), default=100,
        help="annealing sweeps per QUBO restart (default: 100)",
    )
    parser.add_argument(
        "--qubo-restarts", type=_integer_at_least(1), default=3,
        help="simulated-annealing restarts per QUBO batch (default: 3)",
    )
    parser.add_argument("--router", choices=ROUTER_NAMES, help="default: dijkstra")
    parser.add_argument(
        "--compare", action="store_true", help="run all six routing methods"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, help="seeds to compare (requires --compare)"
    )
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    parser.add_argument("--output", type=Path, help="write results to this file")
    parser.add_argument("--quiet", action="store_true", help="hide progress on stderr")
    parser.add_argument(
        "--progress-every", type=_integer_at_least(1), default=50,
        help="progress interval in simulation ticks (default: 50)",
    )
    return parser


def write_results(payload: dict[str, Any], stream: TextIO, output_format: str) -> None:
    if output_format == "json":
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    elif output_format == "csv":
        records = payload["records"] if "records" in payload else [payload]
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    else:
        raise ValueError(f"unsupported output format: {output_format}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seeds is not None and not args.compare:
        parser.error("--seeds requires --compare")
    if args.seeds is not None and args.seed is not None:
        parser.error("use either --seed or --seeds, not both")
    if args.compare and args.router is not None:
        parser.error("--compare runs every router; omit --router")
    if args.seeds is not None and len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain unique values")
    config = ExperimentConfig(
        width=args.width,
        height=args.height,
        agent_count=args.agents,
        seed=args.seed if args.seed is not None else 7,
        max_ticks=args.max_ticks,
        reroute_wait_threshold=args.reroute_wait,
        disaster_profile=args.disaster,
        hazard_weight=args.hazard_weight,
        qubo_batch_size=args.qubo_batch_size,
        qubo_sweeps=args.qubo_sweeps,
        qubo_routes_per_shelter=args.qubo_routes_per_shelter,
        qubo_congestion_weight=args.qubo_congestion_weight,
        qubo_restarts=args.qubo_restarts,
    )

    def report(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    status = None if args.quiet else report
    try:
        if args.compare:
            output = compare_routers(
                config, args.seeds, status=status, progress_interval=args.progress_every
            )
        else:
            output = run_scenario(
                config, args.router or "dijkstra",
                status=status, progress_interval=args.progress_every,
            )
        if args.output is None:
            write_results(output, sys.stdout, args.format)
        else:
            with args.output.open("w", encoding="utf-8", newline="") as stream:
                write_results(output, stream, args.format)
            if status:
                status(f"Results saved to {args.output.resolve()}")
    except OSError as exc:
        parser.exit(1, f"exev: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "exev: interrupted\n")


if __name__ == "__main__":
    main()
