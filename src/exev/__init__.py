"""ExEv: a solver-agnostic evacuation research simulator."""

__version__ = "0.5.0"

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
    RouteAssignmentQUBO,
    RouteOption,
    SimulatedAnnealingQUBOSolver,
    build_route_assignment_qubo,
)
from .simulation import EvacuationSimulation, SimulationConfig, SimulationResult
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
    "ExactQUBOSolver",
    "QUBOModel",
    "QUBOSimulatedAnnealingRouter",
    "QUBOSolution",
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
    "create_router",
    "grid_disaster_schedule",
    "plan_study",
    "run_study",
    "summarize_records",
]
