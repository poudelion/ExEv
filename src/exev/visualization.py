"""Serializable time-series snapshots for the ExEv digital-twin dashboard."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any

from .disasters import grid_disaster_schedule
from .experiments import ExperimentConfig, runtime_metadata
from .models import Agent, AgentStatus
from .routing import create_router
from .scenarios import grid_scenario
from .simulation import EvacuationSimulation, SimulationConfig


def _agent_position(agent: Agent, simulation: EvacuationSimulation) -> tuple[float, float]:
    network = simulation.network
    if agent.edge is None:
        return network.position(agent.current_node) or (0.0, 0.0)
    source, target = agent.edge
    start = network.position(source) or (0.0, 0.0)
    end = network.position(target) or start
    fraction = min(1.0, agent.edge_progress / network.edge(source, target).distance)
    return (
        start[0] + (end[0] - start[0]) * fraction,
        start[1] + (end[1] - start[1]) * fraction,
    )


def simulation_snapshot(
    simulation: EvacuationSimulation, *, max_rendered_agents: int = 1_000
) -> dict[str, Any]:
    """Capture current mutable state without changing the simulation."""
    if max_rendered_agents < 1:
        raise ValueError("max_rendered_agents must be positive")
    counts = Counter(agent.status.value for agent in simulation.agents)
    stride = max(1, (len(simulation.agents) + max_rendered_agents - 1) // max_rendered_agents)
    selected = simulation.agents[::stride][:max_rendered_agents]
    occupancy = simulation.network.occupancy_snapshot()
    return {
        "tick": simulation.tick,
        "counts": {status.value: counts[status.value] for status in AgentStatus},
        "mean_hazard_exposure": (
            sum(agent.hazard_exposure for agent in simulation.agents) / len(simulation.agents)
            if simulation.agents else 0.0
        ),
        "routing_reasons": simulation.routing_reasons,
        "shelters": [
            {
                "node": node,
                "occupants": shelter.occupants,
                "capacity": shelter.capacity,
                "available": shelter.available,
                "full": shelter.available == 0,
            }
            for node, shelter in sorted(simulation.shelters.items())
        ],
        "agents": [
            {"id": agent.id, "x": x, "y": y, "status": agent.status.value}
            for agent in selected
            for x, y in [_agent_position(agent, simulation)]
        ],
        "nodes": [
            {"id": node, "hazard": simulation.network.node_hazard(node)}
            for node in sorted(simulation.network.nodes)
            if simulation.network.node_hazard(node) > 0
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "occupancy": occupancy.get((edge.source, edge.target), 0),
                "capacity": edge.capacity,
                "blocked": edge.blocked,
                "hazard": simulation.network.edge_hazard(edge.source, edge.target),
            }
            for edge in simulation.network.edges
            if edge.blocked
            or occupancy.get((edge.source, edge.target), 0)
            or simulation.network.edge_hazard(edge.source, edge.target) > 0
        ],
    }


def build_visualization_run(
    config: ExperimentConfig,
    algorithm: str,
    *,
    frame_interval: int = 2,
    max_rendered_agents: int = 1_000,
) -> dict[str, Any]:
    """Run a fresh scenario and return topology, frames, and final metrics."""
    if frame_interval < 1:
        raise ValueError("frame_interval must be positive")
    network, agents, shelters = grid_scenario(
        config.width, config.height, config.agent_count, config.seed
    )
    router = create_router(
        algorithm,
        hazard_weight=config.hazard_weight,
        qubo_batch_size=config.qubo_batch_size,
        qubo_routes_per_shelter=config.qubo_routes_per_shelter,
        qubo_congestion_weight=config.qubo_congestion_weight,
        qubo_sweeps=config.qubo_sweeps,
        qubo_restarts=config.qubo_restarts,
        seed=config.seed,
    )
    simulation = EvacuationSimulation(
        network,
        agents,
        shelters,
        router=router,
        config=SimulationConfig(config.max_ticks, config.reroute_wait_threshold),
        disaster_schedule=grid_disaster_schedule(
            config.width, config.height, config.agent_count, config.disaster_profile
        ),
    )
    frames: list[dict[str, Any]] = []

    def capture(current: EvacuationSimulation) -> None:
        if not frames or frames[-1]["tick"] != current.tick:
            frames.append(
                simulation_snapshot(current, max_rendered_agents=max_rendered_agents)
            )

    result = simulation.run(progress=capture, progress_interval=frame_interval)
    return {
        "algorithm": algorithm,
        "topology": {
            "nodes": [
                {"id": node, "x": position[0], "y": position[1]}
                for node in sorted(network.nodes)
                if (position := network.position(node)) is not None
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "capacity": edge.capacity,
                }
                for edge in network.edges
                if edge.source < edge.target
            ],
            "shelters": [
                {"node": node, "capacity": shelter.capacity}
                for node, shelter in sorted(simulation.shelters.items())
            ],
        },
        "frames": frames,
        "result": {**asdict(result), "evacuation_rate": result.evacuation_rate},
    }


def build_dashboard_payload(
    config: ExperimentConfig,
    algorithms: list[str],
    *,
    frame_interval: int = 2,
    max_rendered_agents: int = 1_000,
) -> dict[str, Any]:
    if not 1 <= len(algorithms) <= 2 or len(set(algorithms)) != len(algorithms):
        raise ValueError("choose one or two unique algorithms")
    return {
        "schema_version": 1,
        "metadata": runtime_metadata(),
        "configuration": asdict(config),
        "runs": [
            build_visualization_run(
                config,
                algorithm,
                frame_interval=frame_interval,
                max_rendered_agents=max_rendered_agents,
            )
            for algorithm in algorithms
        ],
    }
