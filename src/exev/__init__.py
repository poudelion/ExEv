"""ExEv: a solver-agnostic evacuation research simulator."""

__version__ = "0.8.0"

from .disasters import (
    DISASTER_PROFILES,
    DisasterSchedule,
    EdgeHazardEvent,
    NodeHazardEvent,
    RoadStatusEvent,
    ShelterCapacityEvent,
    grid_disaster_schedule,
)
from .models import Agent, AgentStatus, Edge, Shelter
from .network import CityNetwork
from .osm import OSMImportStats, OSMScenario, load_osm_scenario
from .flow import MinCostFlowRouter
from .routing import (
    AStarRouter,
    CongestionAwareRouter,
    DijkstraRouter,
    HazardAwareRouter,
    RoutingStats,
    create_router,
)
from .qubo import (
    ExactQUBOSolver,
    QUBOModel,
    QUBOSimulatedAnnealingRouter,
    QUBOSolution,
    QUBOSolver,
    RouteAssignmentQUBO,
    RouteOption,
    SimulatedAnnealingQUBOSolver,
    build_route_assignment_qubo,
)
from .qubo_benchmark import (
    QUBOBackendResult,
    QUBOBenchmarkReport,
    compare_qubo_backends,
)
from .qubo_io import (
    QUBO_FORMAT,
    QUBO_FORMAT_VERSION,
    dumps_qubo,
    loads_qubo,
    qubo_fingerprint,
    qubo_from_dict,
    qubo_to_dict,
    read_qubo,
    write_qubo,
)
from .simulation import EvacuationSimulation, SimulationConfig, SimulationResult
from .visualization import (
    build_dashboard_payload,
    build_osm_dashboard_payload,
    build_visualization_run,
    simulation_snapshot,
)
from .studies import (
    StudyDefinition,
    StudyOutcome,
    StudyRun,
    plan_study,
    run_study,
    summarize_records,
)

__all__ = [
    "Agent",
    "AgentStatus",
    "CityNetwork",
    "DijkstraRouter",
    "DISASTER_PROFILES",
    "DisasterSchedule",
    "EdgeHazardEvent",
    "AStarRouter",
    "CongestionAwareRouter",
    "Edge",
    "EvacuationSimulation",
    "HazardAwareRouter",
    "Shelter",
    "SimulationConfig",
    "SimulationResult",
    "MinCostFlowRouter",
    "NodeHazardEvent",
    "OSMImportStats",
    "OSMScenario",
    "ExactQUBOSolver",
    "QUBOBackendResult",
    "QUBOBenchmarkReport",
    "QUBOModel",
    "QUBO_FORMAT",
    "QUBO_FORMAT_VERSION",
    "QUBOSimulatedAnnealingRouter",
    "QUBOSolution",
    "QUBOSolver",
    "RoadStatusEvent",
    "RouteAssignmentQUBO",
    "RouteOption",
    "RoutingStats",
    "ShelterCapacityEvent",
    "SimulatedAnnealingQUBOSolver",
    "StudyDefinition",
    "StudyOutcome",
    "StudyRun",
    "build_route_assignment_qubo",
    "build_dashboard_payload",
    "build_osm_dashboard_payload",
    "build_visualization_run",
    "compare_qubo_backends",
    "create_router",
    "dumps_qubo",
    "grid_disaster_schedule",
    "loads_qubo",
    "load_osm_scenario",
    "plan_study",
    "qubo_fingerprint",
    "qubo_from_dict",
    "qubo_to_dict",
    "read_qubo",
    "run_study",
    "summarize_records",
    "simulation_snapshot",
    "write_qubo",
]
