"""Run an ExEv experiment on a local OpenStreetMap XML extract."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .experiments import runtime_metadata
from .osm import load_osm_scenario
from .routing import ROUTER_NAMES, create_router
from .simulation import EvacuationSimulation, SimulationConfig


def _nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import an OSM XML map and run an ExEv routing experiment",
        allow_abbrev=False,
    )
    parser.add_argument("--input", type=Path, required=True, help="local .osm XML file")
    parser.add_argument("--shelter-nodes", nargs="+", required=True, help="OSM node IDs")
    parser.add_argument("--origin-nodes", nargs="+", required=True, help="OSM node IDs")
    parser.add_argument("--agents", type=_nonnegative, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--router", choices=ROUTER_NAMES, default="dijkstra")
    parser.add_argument("--max-ticks", type=int, default=10_000)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_ticks < 1:
        parser.error("--max-ticks must be positive")
    if len(set(args.shelter_nodes)) != len(args.shelter_nodes):
        parser.error("--shelter-nodes must be unique")
    if len(set(args.origin_nodes)) != len(args.origin_nodes):
        parser.error("--origin-nodes must be unique")
    try:
        scenario = load_osm_scenario(
            args.input,
            shelter_nodes=args.shelter_nodes,
            origin_nodes=args.origin_nodes,
            agent_count=args.agents,
            seed=args.seed,
        )
        simulation = EvacuationSimulation(
            scenario.network,
            scenario.agents,
            scenario.shelters,
            router=create_router(args.router, seed=args.seed),
            config=SimulationConfig(max_ticks=args.max_ticks),
        )
        result = simulation.run()
        output = {
            "schema_version": 1,
            "metadata": runtime_metadata(),
            "source": str(args.input.resolve()),
            "import": asdict(scenario.stats),
            "configuration": {
                "agent_count": args.agents,
                "seed": args.seed,
                "router": args.router,
                "shelter_nodes": args.shelter_nodes,
                "origin_nodes": args.origin_nodes,
                "distance_unit": "meters",
                "speed_unit": "meters_per_tick",
                "capacity_model": "estimated_from_lanes_times_20",
            },
            "result": {**asdict(result), "evacuation_rate": result.evacuation_rate},
        }
        document = json.dumps(output, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(document, encoding="utf-8")
        else:
            print(document, end="")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"exev-osm: {exc}\n")


if __name__ == "__main__":
    main()
