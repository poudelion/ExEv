"""Command-line interface for reproducible ExEv experiment studies."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .disasters import DISASTER_PROFILES
from .routing import ROUTER_NAMES
from .studies import StudyDefinition, plan_study, run_study


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


def _map_size(value: str) -> tuple[int, int]:
    normalized = value.lower().replace("×", "x")
    try:
        width_text, height_text = normalized.split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("must use WIDTHxHEIGHT, for example 8x8") from exc
    if width < 2 or height < 2:
        raise argparse.ArgumentTypeError("width and height must be at least 2")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a reproducible, resumable ExEv experiment study",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--maps", nargs="+", type=_map_size, default=[(8, 8)],
        metavar="WIDTHxHEIGHT", help="grid sizes to test (default: 8x8)",
    )
    parser.add_argument(
        "--agents", nargs="+", type=_integer_at_least(0), default=[100],
        help="population sizes to test (default: 100)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[7, 8, 9],
        help="scenario seeds (default: 7 8 9)",
    )
    parser.add_argument(
        "--disasters", nargs="+", choices=DISASTER_PROFILES,
        default=["none", "fire"], help="disaster profiles (default: none fire)",
    )
    parser.add_argument(
        "--algorithms", nargs="+", choices=ROUTER_NAMES,
        default=list(ROUTER_NAMES), help="routing algorithms (default: all)",
    )
    parser.add_argument(
        "--hazard-weights", nargs="+", type=_nonnegative_float, default=[1.0],
        help="weights swept for hazard-aware and qubo-sa (default: 1.0)",
    )
    parser.add_argument(
        "--qubo-batch-sizes", nargs="+", type=_integer_at_least(1), default=[8],
    )
    parser.add_argument(
        "--qubo-routes-per-shelter", nargs="+",
        type=_integer_at_least(1), default=[3],
    )
    parser.add_argument(
        "--qubo-congestion-weights", nargs="+",
        type=_nonnegative_float, default=[1.0],
    )
    parser.add_argument(
        "--qubo-sweeps", nargs="+", type=_integer_at_least(1), default=[100],
    )
    parser.add_argument(
        "--qubo-restarts", nargs="+", type=_integer_at_least(1), default=[3],
    )
    parser.add_argument("--max-ticks", type=_integer_at_least(1), default=10_000)
    parser.add_argument("--reroute-wait", type=_integer_at_least(1), default=10)
    parser.add_argument(
        "--progress-every", type=_integer_at_least(1), default=50,
        help="simulation progress interval in ticks (default: 50)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("study-results.csv"),
        help="raw result CSV; summary and manifest use the same stem",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="fail instead of resuming when the raw output already exists",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate and show the expanded run count without simulating",
    )
    parser.add_argument("--quiet", action="store_true", help="hide progress on stderr")
    return parser


def _definition(args: argparse.Namespace) -> StudyDefinition:
    return StudyDefinition(
        map_sizes=tuple(args.maps),
        agent_counts=tuple(args.agents),
        seeds=tuple(args.seeds),
        disaster_profiles=tuple(args.disasters),
        algorithms=tuple(args.algorithms),
        hazard_weights=tuple(args.hazard_weights),
        qubo_batch_sizes=tuple(args.qubo_batch_sizes),
        qubo_routes_per_shelter=tuple(args.qubo_routes_per_shelter),
        qubo_congestion_weights=tuple(args.qubo_congestion_weights),
        qubo_sweeps=tuple(args.qubo_sweeps),
        qubo_restarts=tuple(args.qubo_restarts),
        max_ticks=args.max_ticks,
        reroute_wait_threshold=args.reroute_wait,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        definition = _definition(args)
        plan = plan_study(definition)
        if args.dry_run:
            json.dump(
                {
                    "planned_runs": len(plan),
                    "definition": asdict(definition),
                    "output": str(args.output.resolve()),
                },
                sys.stdout,
                indent=2,
                allow_nan=False,
            )
            sys.stdout.write("\n")
            return

        def report(message: str) -> None:
            print(message, file=sys.stderr, flush=True)

        outcome = run_study(
            definition,
            args.output,
            resume=not args.no_resume,
            status=None if args.quiet else report,
            progress_interval=args.progress_every,
        )
        json.dump(
            {
                "planned_runs": outcome.planned_runs,
                "completed_runs": outcome.completed_runs,
                "skipped_runs": outcome.skipped_runs,
                "raw_results": str(outcome.raw_results),
                "summary_results": str(outcome.summary_results),
                "manifest": str(outcome.manifest),
            },
            sys.stdout,
            indent=2,
            allow_nan=False,
        )
        sys.stdout.write("\n")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"exev-study: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "exev-study: interrupted\n")


if __name__ == "__main__":
    main()
