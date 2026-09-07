"""CLI for benchmarking identical serialized QUBOs across solver backends."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .experiments import runtime_metadata
from .qubo import ExactQUBOSolver, SimulatedAnnealingQUBOSolver
from .qubo_benchmark import compare_qubo_backends
from .qubo_io import read_qubo

BACKEND_NAMES = ("exact", "simulated-annealing")


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


def _binary_sample(value: str) -> tuple[int, ...]:
    normalized = value.replace(",", "").replace(" ", "")
    if any(character not in "01" for character in normalized):
        raise argparse.ArgumentTypeError("must contain only zero and one")
    return tuple(int(character) for character in normalized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare local solver backends on one serialized ExEv QUBO",
        allow_abbrev=False,
    )
    parser.add_argument("--input", type=Path, required=True, help="ExEv QUBO JSON file")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=BACKEND_NAMES,
        default=list(BACKEND_NAMES),
        help="backends to compare (default: exact simulated-annealing)",
    )
    parser.add_argument(
        "--initial-sample",
        type=_binary_sample,
        help="optional warm start such as 0101 or 0,1,0,1",
    )
    parser.add_argument(
        "--exact-max-variables",
        type=_integer_at_least(0),
        default=24,
        help="exact enumeration limit (default: 24)",
    )
    parser.add_argument("--sweeps", type=_integer_at_least(1), default=100)
    parser.add_argument("--restarts", type=_integer_at_least(1), default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, help="write benchmark JSON to this file")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(set(args.backends)) != len(args.backends):
        parser.error("--backends must contain unique values")
    try:
        model = read_qubo(args.input)
        if args.initial_sample is not None:
            model.validate_sample(args.initial_sample)
        available = {
            "exact": ExactQUBOSolver(max_variables=args.exact_max_variables),
            "simulated-annealing": SimulatedAnnealingQUBOSolver(
                sweeps=args.sweeps,
                restarts=args.restarts,
                seed=args.seed,
            ),
        }
        report = compare_qubo_backends(
            model,
            {name: available[name] for name in args.backends},
            initial_sample=args.initial_sample,
        )
        payload = {
            "metadata": runtime_metadata(),
            "input": str(args.input.resolve()),
            **report.as_dict(),
        }
        if args.output is None:
            json.dump(payload, sys.stdout, indent=2, allow_nan=False)
            sys.stdout.write("\n")
        else:
            destination = args.output.resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, allow_nan=False)
                stream.write("\n")
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"exev-qubo: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "exev-qubo: interrupted\n")


if __name__ == "__main__":
    main()
