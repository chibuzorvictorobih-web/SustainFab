"""Exhaustive multi-scenario line optimization with a verified Pareto frontier.

We enumerate every permitted machine-count/mode combination. The performance
model is identical to model.simulate_line, vectorized with NumPy for speed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Sequence

import numpy as np

from .model import (
    MAX_MACHINES,
    OperatingConditions,
    SimulationResult,
    SPEED_MODES,
    Stage,
    downstream_yields,
    simulate_line,
    validate_inputs,
)

Objective = Literal["min_energy_for_target", "max_output_for_budget"]
MAX_CANDIDATES = 600_000


@dataclass(frozen=True)
class StageChoice:
    name: str
    machines: int
    speed_mode: str


@dataclass(frozen=True)
class ParetoPoint:
    good_units: float
    energy_kwh: float


@dataclass(frozen=True)
class OptimizationResult:
    objective: Objective
    feasible: bool
    feasible_count: int
    evaluated_candidates: int
    target_units: float | None
    energy_budget_kwh: float | None
    choices: tuple[StageChoice, ...]
    simulation: SimulationResult
    pareto_frontier: tuple[ParetoPoint, ...]
    status_message: str


def apply_choices(stages: Sequence[Stage], choices: Sequence[StageChoice]) -> tuple[Stage, ...]:
    if len(stages) != len(choices):
        raise ValueError("Each stage must have a matching optimizer choice.")
    if any(s.name != c.name for s, c in zip(stages, choices)):
        raise ValueError("Choice order/names must match the original stages.")
    return tuple(replace(s, machines=c.machines) for s, c in zip(stages, choices))


def _candidate_options(stages: Sequence[Stage], station_delta: int) -> list[list[StageChoice]]:
    result = []
    for stage in stages:
        counts = range(max(1, stage.machines - station_delta), min(MAX_MACHINES, stage.machines + station_delta) + 1)
        result.append([StageChoice(stage.name, count, mode) for count in counts for mode in SPEED_MODES])
    return result


def optimize_line(
    stages: Sequence[Stage],
    conditions: OperatingConditions,
    *,
    objective: Objective = "min_energy_for_target",
    target_fraction: float = 0.95,
    energy_budget_kwh: float | None = None,
    station_delta: int = 1,
) -> OptimizationResult:
    """Find the GLOBAL optimum within the explicitly enumerated search space.

    Candidate choices: each stage gets between baseline -/+ station_delta
    machines (clipped to 1..20), combined with all three operating modes.
    Tie-breaking: lower energy, then higher production for the min-energy goal;
    higher production, then lower energy for the capped-energy goal.
    """
    validate_inputs(stages, conditions)
    if objective not in ("min_energy_for_target", "max_output_for_budget"):
        raise ValueError("Invalid optimization objective.")
    if not 0 <= target_fraction <= 1:
        raise ValueError("Target fraction must be from 0 to 1.")
    if isinstance(station_delta, bool) or not isinstance(station_delta, int) or not 0 <= station_delta <= 2:
        raise ValueError("Station delta must be an integer from 0 to 2.")
    if objective == "max_output_for_budget":
        if energy_budget_kwh is None or not np.isfinite(energy_budget_kwh) or energy_budget_kwh < 0:
            raise ValueError("Enter a finite, nonnegative electricity budget (kWh/day).")

    options = _candidate_options(stages, station_delta)
    sizes = tuple(len(opt) for opt in options)
    candidate_count = int(np.prod(sizes, dtype=np.int64))
    if candidate_count > MAX_CANDIDATES:
        raise ValueError(
            f"This search would require {candidate_count:,} candidates (limit {MAX_CANDIDATES:,}). "
            "Reduce the stations search range or number of stages."
        )
    # Each column is one complete candidate configuration, enumerated without randomness.
    option_indexes = np.indices(sizes, dtype=np.uint8).reshape(len(stages), -1)
    yields = downstream_yields(stages)
    output = np.full(candidate_count, conditions.demand_units, dtype=np.float64)
    stage_params: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []

    for i, stage in enumerate(stages):
        current_options = options[i]
        counts = np.array([choice.machines for choice in current_options], dtype=np.float64)[option_indexes[i]]
        cycles = np.array(
            [stage.cycle_time_min * SPEED_MODES[choice.speed_mode][0] for choice in current_options],
            dtype=np.float64,
        )[option_indexes[i]]
        powers = np.array(
            [stage.run_power_kw * SPEED_MODES[choice.speed_mode][1] for choice in current_options],
            dtype=np.float64,
        )[option_indexes[i]]
        effective_capacity = counts * conditions.shift_hours * stage.availability * 60 / cycles * yields[i]
        np.minimum(output, effective_capacity, out=output)
        stage_params.append((counts, cycles, powers, yields[i]))

    energy = np.zeros(candidate_count, dtype=np.float64)
    for stage, (counts, cycles, powers, downstream) in zip(stages, stage_params):
        available_machine_hours = counts * conditions.shift_hours * stage.availability
        active_machine_hours = output * cycles / (60 * downstream)
        # Mirrors the floating-point safety clamp in simulate_line.
        np.minimum(active_machine_hours, available_machine_hours, out=active_machine_hours)
        energy += active_machine_hours * powers + (available_machine_hours - active_machine_hours) * stage.idle_power_kw

    target_units = None
    budget = None
    if objective == "min_energy_for_target":
        target_units = target_fraction * conditions.demand_units
        eligible = np.flatnonzero(output >= target_units - 1e-7)
        feasible = len(eligible) > 0
        if feasible:
            local_order = np.lexsort((-output[eligible], energy[eligible]))
            selected = int(eligible[local_order[0]])
            message = "Lowest-energy configuration meeting the specified output target."
        else:
            local_order = np.lexsort((energy, -output))
            selected = int(local_order[0])
            message = "Target infeasible in the tested search space. Showing maximum attainable output (not a feasible optimum)."
    else:
        budget = float(energy_budget_kwh)  # validated above
        eligible = np.flatnonzero(energy <= budget + 1e-7)
        feasible = len(eligible) > 0
        if feasible:
            local_order = np.lexsort((energy[eligible], -output[eligible]))
            selected = int(eligible[local_order[0]])
            message = "Highest-output configuration within the specified daily electricity cap."
        else:
            selected = int(np.argmin(energy))
            message = "Electricity cap infeasible in the tested search space. Showing lowest-energy configuration (still over budget)."

    choices = tuple(options[i][int(option_indexes[i, selected])] for i in range(len(stages)))
    configured = apply_choices(stages, choices)
    simulated = simulate_line(configured, conditions, [choice.speed_mode for choice in choices])
    # Verify optimizer's vectorized computation against the independent scalar public API.
    if not (
        np.isclose(simulated.good_units, output[selected], rtol=1e-10, atol=1e-7)
        and np.isclose(simulated.total_energy_kwh, energy[selected], rtol=1e-10, atol=1e-7)
    ):
        raise ArithmeticError("Candidate model and reported solution disagree; optimization halted.")

    # Nondominated points: sort by energy, retain only increasing output.
    sorted_indexes = np.lexsort((-output, energy))
    sorted_output = output[sorted_indexes]
    previous_best = np.r_[-np.inf, np.maximum.accumulate(sorted_output[:-1])]
    frontier_indexes = sorted_indexes[sorted_output > previous_best + 1e-7]
    if len(frontier_indexes) > 240:  # visualization sampling only, NEVER optimizer sampling
        sampled = np.unique(np.linspace(0, len(frontier_indexes) - 1, 240, dtype=int))
        frontier_indexes = frontier_indexes[sampled]
    frontier = tuple(ParetoPoint(float(output[i]), float(energy[i])) for i in frontier_indexes)

    return OptimizationResult(
        objective=objective,
        feasible=bool(feasible),
        feasible_count=len(eligible),
        evaluated_candidates=candidate_count,
        target_units=target_units,
        energy_budget_kwh=budget,
        choices=choices,
        simulation=simulated,
        pareto_frontier=frontier,
        status_message=message,
    )
