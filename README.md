# ExEv

ExEv is a prototype for studying evacuation routing
and comparing classical, QUBO, and eventually quantum/hybrid optimization methods.
Stages 1 through 7 provide a synthetic evacuation simulator, five classical
baselines, deterministic disaster scenarios, an explicit route-assignment QUBO,
an exact reference solver, seeded simulated annealing, a reproducible
experimental pipeline, solver-neutral QUBO benchmarking, and an animated
digital-twin dashboard.

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

An optional virtual environment and standard local installation expose the
shorter `exev`, `exev-study`, `exev-qubo`, and `exev-ui` commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
exev --agents 5000
exev-study --dry-run
exev-qubo --input examples/tiny-qubo.json
exev-ui --agents 500 --disaster fire
```

## Stage 7: digital-twin dashboard

Run two routing strategies on identical fresh scenarios and animate their
results side by side:

```bash
exev-ui \
  --agents 500 \
  --width 10 \
  --height 8 \
  --disaster fire \
  --routers dijkstra hazard-aware
```

Open the printed local URL in a browser. The dashboard displays moving agents,
hazard intensity, closed roads, congestion, shelter locations, evacuation
progress, and final completion time. Use the timeline, play/pause control, and
speed selector to inspect route behavior. Each panel includes a router dropdown, and the control bar includes a hazard
selector. Changing either reruns the selected strategies on the same seeded map
and population. Add `--open` to launch
the browser automatically. For reproducible artifacts, `--export dashboard.json` writes the
same versioned data without starting a server. `--render-agents` limits only the
dots sent to the browser; all agents still contribute to simulation metrics.

Reinstall with `python -m pip install . --force-reinstall` after changing source.
On macOS, an editable `pip install -e .` can produce a hidden `.pth` file that
recent Python versions skip, causing `ModuleNotFoundError` even after a successful
build. The standard install above avoids that platform issue. Source execution
also remains available without reinstalling: use the `PYTHONPATH=src` commands
above or export `PYTHONPATH="$PWD/src"`. These commands assume the current
directory is the project root.

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

## Stage 5: reproducible experiment studies

Use the dedicated study command to expand a scenario grid before spending
compute time:

```bash
PYTHONPATH=src python3 -m exev.study_cli --dry-run --maps 8x8 --agents 100 500 1000 --seeds 7 8 9 10 11 --disasters none road-closure fire flood
```

Run that study and save its artifacts under `results/`:

```bash
PYTHONPATH=src python3 -m exev.study_cli --maps 8x8 --agents 100 500 1000 --seeds 7 8 9 10 11 --disasters none road-closure fire flood --output results/baseline-study.csv --quiet
```

With the default six algorithms, that example expands to 360 simulations.
Start with `--dry-run`, then use a smaller pilot before a long sweep because
QUBO simulated annealing is substantially slower than the classical baselines.
A local installation also exposes the equivalent `exev-study` command.

Every completed simulation is flushed immediately to the raw CSV. Running the
same command again resumes it using deterministic `run_id` values and skips
completed rows. `--no-resume` instead refuses to touch an existing raw CSV.
Changing the study definition or model version requires a new output file; ExEv
rejects rows that do not belong to the current plan.

Each study creates three artifacts:

- `baseline-study.csv`: one auditable row per configuration, algorithm, and seed,
  including the Git commit and whether the working tree was dirty.
- `baseline-study.summary.csv`: replicate count, complete-evacuation count, and
  mean, sample standard deviation, minimum, and maximum for each evaluation metric.
- `baseline-study.manifest.json`: the exact study definition, model version,
  source-control metadata, run counts, and artifact paths.

Parameter sweeps are algorithm-aware. Hazard weights expand `hazard-aware` and
`qubo-sa`; QUBO batch, candidate-route, congestion, sweep, and restart settings
expand only `qubo-sa`. Irrelevant settings do not create duplicate classical
baseline runs. This supports controlled comparisons while retaining every seed
instead of hiding failures inside an average.

## Stage 6A: solver-neutral QUBO experiments

Stage 6 begins with a backend-neutral boundary rather than a hardware-specific
claim. Any local or remote backend can implement
`solve(model, initial_sample)` and return a validated `QUBOSolution`. The
batched evacuation router also accepts an injected backend while retaining
seeded simulated annealing as its dependency-free default.

QUBOs can be written to a versioned, name-based JSON format and reconstructed
without changing their energy function. A SHA-256 fingerprint identifies the
mathematical problem, so different solvers can be shown to have received the
same linear terms, quadratic terms, variable order, and constant.

Compare the included exact and simulated-annealing backends on the example:

```bash
PYTHONPATH=src python3 -m exev.qubo_cli \
  --input examples/tiny-qubo.json \
  --initial-sample 10 \
  --sweeps 100 \
  --restarts 3
```

After installation, `exev-qubo` is equivalent. The report records the QUBO
fingerprint, term counts, sample, recomputed energy, gap from the best observed
energy, iteration counters, and measured solve time. An exact result proves the
optimum only when the model is within the configured enumeration limit, which
defaults to 24 variables. Without exact enumeration, best observed is not a
proof of optimality.

The generic JSON format describes an unconstrained binary objective, so
`constraint_feasible` is null in the CLI report. Programmatic route-assignment
benchmarks can pass `RouteAssignmentQUBO.is_feasible` to
`compare_qubo_backends` and receive an explicit feasibility result. This
validation boundary is where a later D-Wave adapter will report samples and
hardware timing. Stage 6A does not claim quantum execution or advantage.

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
| `p50_evacuation_time`, `p90_evacuation_time`, `p95_evacuation_time` | Median and upper-tail arrival ticks for evacuated agents. |
| `total_waiting_time`, `mean_waiting_time`, `p95_waiting_time` | Aggregate, average, and upper-tail queue delay across all agents. |
| `total_distance_traveled`, `mean_distance_traveled` | Actual distance across all agents, including incomplete trips. |
| `completion_time_gini`, `hazard_exposure_gini` | Dispersion from 0 (equal) toward 1 (unequal). Completion uses the final tick as a censored value for agents not evacuated. |
| `slow_agent_mean_evacuation_time`, `fast_agent_mean_evacuation_time`, `speed_group_evacuation_gap` | Evacuated-only means for the slowest and fastest speed groups and slow-minus-fast gap; null when a group has no evacuated member. |
| `peak_congestion` | Highest observed occupancy/capacity ratio on any directed road. |
| `total_edge_occupancy_ticks`, `full_edge_ticks`, `mean_edge_utilization` | Network load: summed road occupancy, number of directed edge/tick observations at capacity, and average occupancy/capacity across all directed roads and ticks. |
| `wall_seconds` | Wall-clock time inside the simulation run, including preparation and progress callbacks; excludes scenario construction and output serialization. |
| `preparation_seconds` | Time spent preparing the router, including the initial min-cost-flow solve. |
| `routing_seconds` | Time spent in routing calls and per-tick routing updates, excluding initial preparation. |
| `route_calls`, `route_searches`, `cache_hits`, `expanded_nodes` | Routing activity counters for understanding repeated requests, reuse, and search work. |
| `total_hazard_exposure`, `mean_hazard_exposure`, `p95_hazard_exposure`, `max_hazard_exposure` | Abstract cumulative exposure across all agents and its per-agent summary. |
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

Stage 5 retains raw seed-level outcomes and adds aggregate summaries; use both.
Averages alone can hide incomplete evacuations, tail delays, or unequal exposure.
Peak road load is still a coarse indicator and often reaches 1.0, so interpret it
alongside edge utilization, full-edge ticks, waiting time, and completion rate.

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
hazard-aware routing. Stage 4 adds batched QUBO assignment and annealing
diagnostics. Stage 5 adds resumable study sweeps, aggregate statistics, fairness
metrics, and richer network-load measures. Stage 6A adds portable QUBOs,
interchangeable solver backends, fingerprints, and backend comparisons. Exported
runs identify this model as `stage7-digital-twin-v1`; results from older
model versions should remain labeled separately rather than being pooled.

## Tests and package layout

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The tests cover movement and capacity behavior, optimal shortest-path, flow, and
small QUBO assignments, annealing reproducibility, routing cache validity,
scheduled disasters, hazard exposure, reproducible comparisons, study planning and
resume behavior, QUBO serialization, backend validation, CLI parsing, and
separation of progress from result output.

```text
pyproject.toml             Python package metadata and console command
src/exev/__init__.py       public classes and source version
src/exev/models.py         agents, edges, shelters, and statuses
src/exev/network.py        graph, geometry, occupancy, and topology changes
src/exev/disasters.py      event scheduler and synthetic disaster profiles
src/exev/routing.py        Dijkstra, A*, congestion-aware routing, and factory
src/exev/flow.py           min-cost-flow assignment and routing
src/exev/qubo.py           QUBO model, solver protocol, solvers, and QUBO router
src/exev/qubo_io.py        versioned QUBO JSON and stable fingerprints
src/exev/qubo_benchmark.py identical-model backend comparisons and validation
src/exev/qubo_cli.py       serialized-QUBO comparison command
src/exev/visualization.py simulation frame capture and dashboard payloads
src/exev/ui_cli.py         local digital-twin server and export command
src/exev/web/              responsive canvas dashboard assets
src/exev/simulation.py     movement engine, progress callbacks, and metrics
src/exev/scenarios.py      deterministic synthetic grid scenarios
src/exev/experiments.py    fresh-state comparisons and metadata
src/exev/studies.py        study expansion, resume, summaries, and manifests
src/exev/study_cli.py      study command and parameter-sweep options
src/exev/cli.py            single-run and comparison command output
examples/tiny-qubo.json     portable two-variable QUBO example
tests/                    simulation, routing, flow, and CLI/experiment checks
```

## Roadmap

1. Simulation foundation: implemented.
2. Classical baselines and an initial comparison runner: implemented.
3. Dynamic disasters: deterministic fire/flood stress tests, road closures,
   shelter changes, exposure metrics, and rerouting: implemented.
4. QUBO optimization: route assignment, exact validation, and seeded classical
   simulated annealing: implemented.
5. Experimental framework: reproducible, resumable scenario sweeps, aggregate
   summaries, tail metrics, fairness, and richer congestion measures: implemented.
6. Quantum/hybrid: solver abstraction, QUBO interchange, and controlled local
   backend comparisons implemented; external hardware adapter remains next.
7. Digital-twin UI: animated agents, hazards, congestion, shelters, playback,
   and side-by-side algorithm comparisons: implemented for synthetic maps.
8. Real data: OpenStreetMap and public hazard/evacuation datasets.

The current synthetic simulator is a research prototype and has not been
validated for operational evacuation decisions.
