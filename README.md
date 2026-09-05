# EvacSim

EvacSim is a research-oriented evacuation simulator for comparing classical,
QUBO, and eventually quantum/hybrid optimization methods under identical
dynamic-disaster scenarios.

## Stage 1: simulation foundation

The current implementation provides:

- directed or bidirectional graph-based city/campus networks;
- thousands of independently moving agents with heterogeneous speeds;
- edge capacity constraints and deterministic congestion slowdown;
- capacity-limited shelters;
- blocked roads and automatic route avoidance;
- a discrete-time simulation engine and structured result metrics;
- a routing strategy protocol for adding Stage 2 algorithms cleanly; and
- deterministic, seeded synthetic grid scenarios.

The built-in shortest-distance router exists to make Stage 1 executable. It is
not yet one of the measured Stage 2 benchmark implementations.

## Run it

Python 3.11 or later is required. No third-party runtime dependency is needed.

```bash
python -m pip install -e .
evacsim --agents 5000 --width 12 --height 12 --seed 7
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## Model semantics

One simulation tick has an abstract duration. Edge `distance`, agent `speed`,
and edge `speed_limit` therefore use compatible abstract units. Agents enter an
edge only when its occupancy is below capacity. Their movement rate decreases
as the capacity ratio rises:

```text
effective_speed = min(agent_speed, speed_limit) / (1 + 2 * load^2)
load = occupancy / capacity
```

Closures stop new entries but do not remove agents already traveling on an
edge. That policy is explicit so a later hazard model can distinguish a closed
road from an immediately lethal or impassable road.

## Roadmap

1. **Simulation foundation** - implemented in this release.
2. **Classical baselines** - Dijkstra, A*, congestion-aware routing, and
   min-cost flow behind the `RoutingStrategy` interface.
3. **Dynamic disasters** - scheduled fire, flood, road, and shelter events.
4. **Optimization** - route assignment as QUBO with classical QUBO solvers.
5. **Experiments** - reproducible scenario sweeps, metrics, and exports.
6. **Quantum/hybrid** - compatible subproblems on available hardware.
7. **Digital-twin UI** - animated state and side-by-side comparisons.
8. **Real data** - OpenStreetMap and public hazard/evacuation datasets.

## Package layout

```text
src/evacsim/models.py       agents, edges, shelters, and statuses
src/evacsim/network.py      capacity-aware mutable graph
src/evacsim/routing.py      routing interface and bootstrap router
src/evacsim/simulation.py   discrete-time engine and result metrics
src/evacsim/scenarios.py    reproducible synthetic scenarios
src/evacsim/cli.py          command-line demo
tests/                      behavioral and scale tests
```

