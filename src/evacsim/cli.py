from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .scenarios import grid_scenario
from .simulation import EvacuationSimulation, SimulationConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an EvacSim Stage 1 grid scenario")
    parser.add_argument("--agents", type=int, default=5_000)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--height", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-ticks", type=int, default=10_000)
    args = parser.parse_args()
    network, agents, shelters = grid_scenario(args.width, args.height, args.agents, args.seed)
    result = EvacuationSimulation(
        network, agents, shelters, config=SimulationConfig(max_ticks=args.max_ticks)
    ).run()
    output = asdict(result)
    output["evacuation_rate"] = result.evacuation_rate
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

