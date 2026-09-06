# ExEv

ExEv is a prototype for studying evacuation routing
and comparing classical, QUBO, and eventually quantum/hybrid optimization methods.
Stages 1 through 4 provide a synthetic evacuation simulator, five classical
baselines, deterministic disaster scenarios, an explicit route-assignment QUBO,
an exact reference solver, and seeded simulated annealing.

## Run a simulation

Python 3.11 or newer is required. There are no third-party runtime dependencies.
From the project directory, this source command works without installation:

```bash
PYTHONPATH=src python3 -m exev.cli --agents 5000 --width 12 --height 12
```

Select another algorithm:

```bash
PYTHONPATH=src python3 -m exev.cli --agents 10000 --width 20 --height 15 --router congestion-aware
```

Add a dynamic disaster profile:

```bash
PYTHONPATH=src python3 -m exev.cli --agents 5000 --width 12 --height 12 --disaster fire
```

The full option is `--agents`; the previous accidental abbreviation `--agent`
is now rejected. The simulator immediately prints a preparation message, then
reports progress every 50 ticks. Progress goes to **stderr**, while the final
JSON goes to **stdout**, so redirected results remain valid JSON.

```bash
PYTHONPATH=src python3 -m exev.cli --agents 1000 --progress-every 20
PYTHONPATH=src python3 -m exev.cli --agents 1000 --quiet
```

An optional virtual environment and editable installation expose the shorter
`exev` command:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
exev --agents 5000
```

If an existing environment reports `ModuleNotFoundError`, use the source command
above, or set `export PYTHONPATH="$PWD/src"` in that terminal before running
`exev`. No changes to the system Python or the existing environment are needed
for source execution. These commands assume the current directory is the project
root.

## Compare routing methods

```bash
PYTHONPATH=src python3 -m exev.cli --compare --agents 1000 --width 8 --height 8 --seed 7
```

Run several seeds and export a table:

```bash
PYTHONPATH=src python3 -m exev.cli --compare --agents 1000 --width 8 --height 8 --seeds 7 8 9 --format csv --output comparison.csv --quiet
```

`--compare` runs all six methods for each seed. Every run reconstructs the
network, agents, shelters, and router from scratch. Therefore each method sees
the same starting population and speeds for a seed, and no run inherits another
run's occupied roads, filled shelters, or routing caches.

| CLI name | Routing objective and implementation |
| --- | --- |
| `dijkstra` | Minimum distance to any shelter with space. A reverse, multi-source Dijkstra tree supplies routes for all origins; the tree is reused until topology or available shelter destinations change. |
| `astar` | Minimum distance to any shelter with space. A* searches from each distinct origin and caches the result. A scaled Euclidean heuristic supplies a safe lower bound; missing geometry falls back to a zero heuristic. |
| `congestion-aware` | Estimated travel time using an occupancy snapshot, road speed limits, and the agent's speed. Reverse trees are shared by speed and destination set within each tick. Agents reconsider routes at intersections and after waiting. |
| `hazard-aware` | Current estimated travel time plus a configurable exposure penalty. It includes road intensity during predicted travel and one tick of intensity at the target node, and replans as observed hazards change. |
| `min-cost-flow` | Global static assignment minimizing total distance subject to shelter capacities, with maximum feasible evacuation as the first objective. The resulting shelter assignments guide each agent's route. |
| `qubo-sa` | Hybrid batched route and shelter assignment expressed as a QUBO and solved by seeded classical simulated annealing. The objective combines route cost, current hazard exposure, and shared-road overlap while enforcing one choice per agent and shelter capacities. |

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

## Stage 4: QUBO route assignment

`qubo-sa` creates one binary variable for each agent/route option. By default,
each agent receives up to three lowest-cost simple paths per reachable shelter;
an additional dummy option means "unassigned." Binary slack variables encode
unused shelter capacity. For route
variables `x[i,r]` and slack variables `z[s,k]`, the minimized energy is:

```text
E(x, z) =
    sum(route_cost[i,r] * x[i,r])
  + congestion_weight * sum(shared_edge_pairs)
  + P * sum_i (sum_r x[i,r] - 1)^2
  + P * sum_s (assigned_to_s + sum_k weight[s,k] * z[s,k] - capacity[s])^2
```

The first constraint selects exactly one real or dummy option for every agent.
The second prevents assignments beyond shelter capacity. `P` is chosen above a
conservative objective bound. Route cost is distance plus weighted exposure at
the conditions observed during preparation. The congestion term penalizes pairs
of selected routes that share directed roads.

Large populations are decomposed into deterministic batches because a single
population-wide QUBO grows too quickly. Earlier batches reserve shelter capacity
before later batches are built. This is a hybrid decomposition and can miss a
global optimum; `--qubo-batch-size` controls that tradeoff.
`--qubo-routes-per-shelter` controls the number of candidate paths and therefore
both solution diversity and QUBO size. The simulator still
enforces simultaneous road occupancy each tick. Shared-edge overlap is only a
soft routing proxy, not a lifetime road-capacity constraint.

The included exact solver exhaustively validates QUBOs of at most 24 variables.
The production `qubo-sa` path uses seeded classical simulated annealing and does
not guarantee the optimum. It starts from a feasible greedy assignment and
falls back to that assignment if annealing returns an infeasible sample.

```bash
PYTHONPATH=src python3 -m exev.cli \
  --router qubo-sa \
  --agents 1000 \
  --width 8 \
  --height 8 \
  --qubo-routes-per-shelter 3 \
  --disaster fire \
  --qubo-batch-size 8 \
  --qubo-sweeps 100 \
  --qubo-restarts 3
```

## Dynamic disasters

Use `--disaster` with one of four reproducible profiles:

| Profile | Behavior |
| --- | --- |
| `none` | Static Stage 2 network, retained as the control condition. |
| `road-closure` | At tick 10, every crossing in a central corridor except one bottleneck closes; the crossings reopen at tick 30. |
| `fire` | A synthetic hazard spreads from the upper-left; node and road exposure increase while corridor crossings close progressively from top to bottom. |
| `flood` | Flood exposure rises row by row, flooded corridor crossings close, and the lower shelter is unavailable until tick 30. |

Compare all routers under the same flood timeline:

```bash
PYTHONPATH=src python3 -m exev.cli --compare --agents 1000 --width 8 --height 8 --disaster flood --quiet
```

Events apply at the beginning of their scheduled tick. New travelers cannot
enter a road after it closes, while travelers already on it finish crossing.
Waiting agents replan when a closure invalidates their route. If no route is
currently available but future events remain, an agent waits instead of being
declared permanently stranded. Shelter capacity cannot be reduced below its
current occupancy.

Hazard intensity is an abstract non-negative value accumulated once per tick at
an agent's waiting node or occupied road. Dijkstra, A*, congestion-aware, and
min-cost flow do not include it in their objective. `hazard-aware` adds current
predicted exposure to estimated travel time. `qubo-sa` includes exposure observed
during its initial assignment. The default weight is `1.0`; sweep
the tradeoff with `--hazard-weight`. It reacts to current conditions but does
not forecast future spread.

```bash
PYTHONPATH=src python3 -m exev.cli --router hazard-aware --disaster fire --hazard-weight 1.0
```

The fire and flood profiles are deterministic stress tests, not calibrated
physical models or forecasts.

Min-cost flow remains the deliberately static Stage 2 assignment baseline. It
can find a new path to an assigned shelter after a road closure, but it does not
reassign agents when shelter capacity changes. A flood run may therefore expose
this limitation by leaving initially unassigned agents stranded after the final
event. A later dynamic-flow baseline can address that separately.

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
| `total_hazard_exposure`, `mean_hazard_exposure`, `max_hazard_exposure` | Abstract cumulative exposure across all agents and its per-agent summary. |
| `qubo_batches`, `qubo_total_variables`, `qubo_max_variables` | Number and size of QUBO subproblems; zero for other routers. |
| `qubo_energy` | Sum of selected batch energies. This diagnostic includes penalty constants and should not be compared across different QUBO configurations. |
| `qubo_annealing_iterations`, `qubo_accepted_moves`, `qubo_fallback_batches` | Simulated-annealing work and the number of batches that used the feasible warm-start fallback. |
| `qubo_planned_flow`, `qubo_planned_cost` | Agents assigned a real shelter and total route cost at preparation time. |
| `events_processed` | Scheduled disaster events applied before the run ended. Events after every agent has reached a terminal state are not processed. |

Search counters measure road shortest-path work, including preparation. They do
not count the flow solver's internal residual-network iterations. Cache hits can
also occur during preparation; they need not sum with searches to route calls.

`wall_seconds` already includes preparation and routing; do not add those fields
to it. Use `--quiet` for timing comparisons to avoid terminal progress overhead.
The run configuration records width, height, population, seed, maximum ticks,
rerouting threshold, disaster profile, hazard weight, and QUBO controls alongside Python,
platform, package, and model versions.
Timings depend on machine load and vary across repeated runs; outcome metrics
and search counters are reproducible for the same configuration and code.

`--max-ticks` defaults to 10000. `--reroute-wait` requests another route after 10
queue ticks since the previous route request or road entry by default. Dimensions
must be at least 2, agents must be nonnegative, and tick limits
and progress intervals must be positive. Use either `--seed` or `--seeds`;
`--seeds` requires `--compare`. The comparison runner retains every seed's result
without averaging away failed or incomplete evacuations.

These exports are an initial benchmark runner. The broader Stage 5 framework
will add scenario sweeps, uncertainty analysis, fairness, and richer congestion
measures. Peak road load alone is a coarse congestion indicator and often
reaches 1.0 across methods.

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
that road finish crossing it. The default `none` profile starts with no blocked
roads or hazards. Custom schedules can change directed or bidirectional road
status, node and edge hazard intensity, and shelter capacity. A stranded outcome
means no admissible route remains after the final scheduled event.

Stage 2 corrects the earlier within-tick movement order effect by using a shared
occupancy snapshot, and improves route invalidation when shelters fill. Cached
algorithms may also choose different tied routes. Old Stage 1 output numbers
therefore should not be compared directly with Stage 2 as evidence of algorithmic
improvement. Stage 3 retains the synchronous movement rules and adds beginning-
of-tick events and exposure accounting. Stage 3 v2 strengthens closures and adds
hazard-aware routing. Stage 4 adds batched QUBO assignment and annealing diagnostics.
Exported runs identify this model as `stage4-qubo-v1`; results from older model
versions should remain labeled separately rather than being pooled.

## Tests and package layout

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The tests cover movement and capacity behavior, optimal shortest-path, flow, and
small QUBO assignments, annealing reproducibility, routing cache validity,
scheduled disasters, hazard exposure, reproducible comparisons, CLI validation,
export parsing, and separation of progress from result output.

```text
pyproject.toml             Python package metadata and console command
src/exev/__init__.py       public classes and source version
src/exev/models.py         agents, edges, shelters, and statuses
src/exev/network.py        graph, geometry, occupancy, and topology changes
src/exev/disasters.py      event scheduler and synthetic disaster profiles
src/exev/routing.py        Dijkstra, A*, congestion-aware routing, and factory
src/exev/flow.py           min-cost-flow assignment and routing
src/exev/qubo.py           QUBO model, exact/annealing solvers, and QUBO router
src/exev/simulation.py     movement engine, progress callbacks, and metrics
src/exev/scenarios.py      deterministic synthetic grid scenarios
src/exev/experiments.py    fresh-state comparisons and metadata
src/exev/cli.py            command options, progress, JSON and CSV output
tests/                    simulation, routing, flow, and CLI/experiment checks
```

## Roadmap

1. Simulation foundation: implemented.
2. Classical baselines and an initial comparison runner: implemented.
3. Dynamic disasters: deterministic fire/flood stress tests, road closures,
   shelter changes, exposure metrics, and rerouting: implemented.
4. QUBO optimization: route assignment, exact validation, and seeded classical
   simulated annealing: implemented.
5. Experimental framework: automated scenario sweeps and richer evaluation.
6. Quantum/hybrid: compatible subproblems on available hardware.
7. Digital-twin UI: animated agents, hazards, and algorithm comparisons.
8. Real data: OpenStreetMap and public hazard/evacuation datasets.

The current synthetic simulator is a research prototype and has not been
validated for operational evacuation decisions.
