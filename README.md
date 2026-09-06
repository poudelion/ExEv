# EvacSim

EvacSim is an undergraduate research prototype for studying evacuation routing
and comparing classical, QUBO, and eventually quantum/hybrid optimization methods.
Stages 1 and 2 provide a synthetic evacuation simulator and four classical
baselines. The current experiments use static networks; evolving disasters and
QUBO solvers are later stages.

## Run a simulation

Python 3.11 or newer is required. There are no third-party runtime dependencies.
From the project directory, this source command works without installation:

```bash
PYTHONPATH=src python3 -m evacsim.cli --agents 5000 --width 12 --height 12
```

Select another algorithm:

```bash
PYTHONPATH=src python3 -m evacsim.cli --agents 10000 --width 20 --height 15 --router congestion-aware
```

The full option is `--agents`; the previous accidental abbreviation `--agent`
is now rejected. The simulator immediately prints a preparation message, then
reports progress every 50 ticks. Progress goes to **stderr**, while the final
JSON goes to **stdout**, so redirected results remain valid JSON.

```bash
PYTHONPATH=src python3 -m evacsim.cli --agents 1000 --progress-every 20
PYTHONPATH=src python3 -m evacsim.cli --agents 1000 --quiet
```

An optional virtual environment and editable installation expose the shorter
`evacsim` command:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
evacsim --agents 5000
```

If an existing environment reports `ModuleNotFoundError`, use the source command
above, or set `export PYTHONPATH="$PWD/src"` in that terminal before running
`evacsim`. No changes to the system Python or the existing environment are needed
for source execution. These commands assume the current directory is the project
root.

## Compare the classical baselines

```bash
PYTHONPATH=src python3 -m evacsim.cli --compare --agents 1000 --width 8 --height 8 --seed 7
```

Run several seeds and export a table:

```bash
PYTHONPATH=src python3 -m evacsim.cli --compare --agents 1000 --width 8 --height 8 --seeds 7 8 9 --format csv --output comparison.csv --quiet
```

`--compare` runs all four methods for each seed. Every run reconstructs the
network, agents, shelters, and router from scratch. Therefore each method sees
the same starting population and speeds for a seed, and no run inherits another
run's occupied roads, filled shelters, or routing caches.

| CLI name | Routing objective and implementation |
| --- | --- |
| `dijkstra` | Minimum distance to any shelter with space. A reverse, multi-source Dijkstra tree supplies routes for all origins; the tree is reused until topology or available shelter destinations change. |
| `astar` | Minimum distance to any shelter with space. A* searches from each distinct origin and caches the result. A scaled Euclidean heuristic supplies a safe lower bound; missing geometry falls back to a zero heuristic. |
| `congestion-aware` | Estimated travel time using an occupancy snapshot, road speed limits, and the agent's speed. Reverse trees are shared by speed and destination set within each tick. Agents reconsider routes at intersections and after waiting. |
| `min-cost-flow` | Global static assignment minimizing total distance subject to shelter capacities, with maximum feasible evacuation as the first objective. The resulting shelter assignments guide each agent's route. |

Min-cost flow uses a residual network with supply at origins and capacity at
shelters. Road capacity in this simulator means **simultaneous occupancy**,
not a lifetime limit on the number of agents using a road. Consequently the
static flow assignment does not impose occupancy limits as total-flow limits;
the movement simulation still enforces them on every road. This baseline
optimizes static distance and shelter assignment, not evacuation completion time
or a schedule through time. A time-expanded flow model would be a separate
future baseline.

Shelter slots are reserved by the initial assignment; agents enter only their
assigned shelter. Road changes can trigger a new path to that destination, but
this static baseline does not recompute shelter assignments.

Caching is part of the measured implementations. Dijkstra's shared reverse tree
and A*'s per-origin search have different costs, so wall-clock measurements are
comparisons of these implementations, not universal claims about the algorithms.
Equal-distance paths may differ between algorithms and create different traffic
patterns; two distance-optimal routers need not produce identical evacuation
times. Congestion-aware routing estimates current conditions and can herd agents
onto the same route; it does not forecast downstream queues or guarantee better
evacuation times.

## Results and experiment controls

Single-run JSON keeps the original result fields at the top level. Comparison
JSON has `metadata`, `configuration`, `seeds`, `algorithms`, and a `records` list.
CSV contains one flat row per algorithm/seed pair, including configuration and
version information, so each row can be interpreted independently.

| Field | Meaning |
| --- | --- |
| `elapsed_ticks` | Number of simulated ticks executed. |
| `evacuated`, `stranded`, `unfinished` | Agents in each outcome group when execution stops. Unfinished agents can remain when `--max-ticks` is reached. |
| `evacuation_rate` | Evacuated / total agents; defined as 1.0 for an empty population. |
| `mean_evacuation_time`, `max_evacuation_time` | Arrival tick statistics for evacuated agents only; null when no one evacuated. |
| `mean_waiting_time` | Queue ticks averaged over all agents. |
| `mean_distance_traveled` | Actual traveled distance averaged over all agents, including incomplete trips. |
| `peak_congestion` | Highest observed occupancy/capacity ratio on any directed road. |
| `wall_seconds` | Wall-clock time inside the simulation run, including preparation and progress callbacks; excludes scenario construction and output serialization. |
| `preparation_seconds` | Time spent preparing the router, including the initial min-cost-flow solve. |
| `routing_seconds` | Time spent in routing calls and per-tick routing updates, excluding initial preparation. |
| `route_calls`, `route_searches`, `cache_hits`, `expanded_nodes` | Routing activity counters for understanding repeated requests, reuse, and search work. |

Search counters measure road shortest-path work, including preparation. They do
not count the flow solver's internal residual-network iterations. Cache hits can
also occur during preparation; they need not sum with searches to route calls.

`wall_seconds` already includes preparation and routing; do not add those fields
to it. Use `--quiet` for timing comparisons to avoid terminal progress overhead.
The run configuration records width, height, population, seed, maximum ticks,
and rerouting threshold, alongside Python, platform, package, and model versions.
Timings depend on machine load and vary across repeated runs; outcome metrics
and search counters are reproducible for the same configuration and code.

`--max-ticks` defaults to 10000. `--reroute-wait` requests another route after 10
queue ticks since the previous route request or road entry by default. Dimensions
must be at least 2, agents must be nonnegative, and tick limits
and progress intervals must be positive. Use either `--seed` or `--seeds`;
`--seeds` requires `--compare`. The comparison runner retains every seed's result
without averaging away failed or incomplete evacuations.

These exports are an initial benchmark runner. The broader Stage 5 framework
will add scenario sweeps, uncertainty analysis, hazard exposure, fairness,
and richer congestion measures. Peak road load alone is a coarse congestion
indicator and often reaches 1.0 across methods.

## Model assumptions

The scenario generator creates a rectangular grid with two shelters on the
right boundary, agents starting in the left portion, and seeded speed choices.
Roads are directed; a bidirectional connection creates two independent edges
with their own occupancy limits. Shelter capacity is finite. Agents queue at
nodes when the next road is full. Nodes themselves have no capacity limit.

One tick has an abstract duration, and distance and speed use compatible
abstract units. Movement follows:

```text
load = occupancy / road_capacity
effective_speed = min(agent_speed, road_speed_limit) / (1 + 2 * load^2)
```

The congestion function is a modeling assumption, not a calibrated traffic or
pedestrian model. All travelers use the same occupancy snapshot for a movement
phase. An agent can finish at most one edge during a tick; leftover movement
does not carry into another edge. Arrivals and road entry are processed after
movement, using a rotating agent priority at queues. This reduces permanent
priority for low IDs but is not a validated fairness policy.

`evacuated_at` uses zero-based tick labels. Thus an evacuation at tick 234 can
appear in a run with 235 elapsed ticks. There is no conversion from these values
to real minutes or meters yet.

Blocked roads reject new entries and routing avoids them; travelers already on
that road finish crossing it. The default CLI grid starts with no blocked roads.
The API supports setting closures for custom scenarios, but there is no fire,
flood, or scheduled disaster-event engine yet. A stranded outcome means no
admissible route is available in the current model and is terminal for that run.

Stage 2 corrects the earlier within-tick movement order effect by using a shared
occupancy snapshot, and improves route invalidation when shelters fill. Cached
algorithms may also choose different tied routes. Old Stage 1 output numbers
therefore should not be compared directly with Stage 2 as evidence of algorithmic
improvement. Exported runs identify this model as `stage2-synchronous-v1`.

## Tests and package layout

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The tests cover movement and capacity behavior, optimal shortest-path and flow
assignments on small graphs, routing cache validity, reproducible comparisons,
CLI validation, export parsing, and separation of progress from result output.

```text
pyproject.toml             Python package metadata and console command
src/evacsim/__init__.py    public classes and source version
src/evacsim/models.py      agents, edges, shelters, and statuses
src/evacsim/network.py     graph, geometry, occupancy, and topology changes
src/evacsim/routing.py     Dijkstra, A*, congestion-aware routing, and factory
src/evacsim/flow.py        min-cost-flow assignment and routing
src/evacsim/simulation.py  movement engine, progress callbacks, and metrics
src/evacsim/scenarios.py   deterministic synthetic grid scenarios
src/evacsim/experiments.py fresh-state comparisons and metadata
src/evacsim/cli.py         command options, progress, JSON and CSV output
tests/                    simulation, routing, flow, and CLI/experiment checks
```

## Roadmap

1. Simulation foundation: implemented.
2. Classical baselines and an initial comparison runner: implemented.
3. Dynamic disasters: fire spread, flooding, road closures, and shelter changes.
4. QUBO optimization: route-assignment formulations and classical QUBO solvers.
5. Experimental framework: automated scenario sweeps and richer evaluation.
6. Quantum/hybrid: compatible subproblems on available hardware.
7. Digital-twin UI: animated agents, hazards, and algorithm comparisons.
8. Real data: OpenStreetMap and public hazard/evacuation datasets.

The current synthetic simulator is a research prototype and has not been
validated for operational evacuation decisions.
