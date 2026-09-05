from __future__ import annotations

import random

from .models import Agent, Shelter
from .network import CityNetwork


def grid_scenario(
    width: int = 12,
    height: int = 12,
    agent_count: int = 5_000,
    seed: int = 7,
) -> tuple[CityNetwork, list[Agent], list[Shelter]]:
    if width < 2 or height < 2:
        raise ValueError("grid dimensions must be at least 2")
    rng = random.Random(seed)
    network = CityNetwork()
    for y in range(height):
        for x in range(width):
            node = f"{x},{y}"
            if x + 1 < width:
                network.add_edge(node, f"{x + 1},{y}", distance=1.0, capacity=40)
            if y + 1 < height:
                network.add_edge(node, f"{x},{y + 1}", distance=1.0, capacity=40)

    shelter_nodes = [f"{width - 1},0", f"{width - 1},{height - 1}"]
    capacity = (agent_count + len(shelter_nodes) - 1) // len(shelter_nodes)
    shelters = [Shelter(node, capacity) for node in shelter_nodes]
    origins = [
        f"{x},{y}"
        for y in range(height)
        for x in range(max(1, width // 2))
        if f"{x},{y}" not in shelter_nodes
    ]
    speed_choices = (0.65, 0.8, 1.0, 1.15)
    agents = [
        Agent(i, rng.choice(origins), speed=rng.choice(speed_choices))
        for i in range(agent_count)
    ]
    return network, agents, shelters

