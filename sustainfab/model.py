"""Transparent, deterministic manufacturing-line throughput and energy model.

All output quantities are expected (possibly fractional) units per scheduled day.
No stochastic failures, queueing, changeovers, or embodied emissions are simulated.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence


# (processing-time multiplier, operating-power multiplier).
# These are scenario assumptions, NOT empirically validated motor curves.
SPEED_MODES: dict[str, tuple[float, float]] = {
    "Eco": (1.18, 0.78),
    "Standard": (1.00, 1.00),
    "Boost": (0.82, 1.32),
}

MAX_STAGES = 6
MAX_MACHINES = 20


@dataclass(frozen=True)
class Stage:
    name: str
    machines: int
    cycle_time_min: float
    availability: float
    yield_rate: float
    run_power_kw: float
    idle_power_kw: float


@dataclass(frozen=True)
class OperatingConditions:
    shift_hours: float = 8.0
    demand_units: float = 400.0
    electricity_tariff: float = 0.15
    emission_factor: float = 0.45


@dataclass(frozen=True)
class StageMetrics:
    name: str
    machines: int
    speed_mode: str
    capacity_input_units: float
    good_capacity_units: float
    processed_input_units: float
    scrapped_units: float
    active_machine_hours: float
    available_machine_hours: float
    utilization_pct: float
    energy_kwh: float


@dataclass(frozen=True)
class SimulationResult:
    good_units: float
    maximum_good_units: float
    fulfillment_pct: float
    total_energy_kwh: float
    emissions_kg: float
    electricity_cost: float
    kwh_per_good_unit: float | None
    bottleneck_stage: str
    stage_metrics: tuple[StageMetrics, ...]


def _finite_number(value: float, description: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{description} must be a finite number.")


def validate_inputs(stages: Sequence[Stage], conditions: OperatingConditions) -> None:
    if not stages:
        raise ValueError("Add at least one manufacturing stage.")
    if len(stages) > MAX_STAGES:
        raise ValueError(f"A maximum of {MAX_STAGES} stages is supported to keep exhaustive optimization responsive.")

    names: set[str] = set()
    for i, stage in enumerate(stages, 1):
        label = f"Stage {i}"
        if not isinstance(stage.name, str) or not stage.name.strip():
            raise ValueError(f"{label}: enter a non-empty stage name.")
        if stage.name.strip().lower() in names:
            raise ValueError(f"Duplicate stage name: {stage.name!r}.")
        names.add(stage.name.strip().lower())
        if isinstance(stage.machines, bool) or not isinstance(stage.machines, int) or not (1 <= stage.machines <= MAX_MACHINES):
            raise ValueError(f"{label}: machine count must be an integer from 1 to {MAX_MACHINES}.")
        for name, value in (
            ("cycle time", stage.cycle_time_min),
            ("availability", stage.availability),
            ("yield", stage.yield_rate),
            ("running power", stage.run_power_kw),
            ("idle power", stage.idle_power_kw),
        ):
            _finite_number(value, f"{label} {name}")
        if stage.cycle_time_min <= 0:
            raise ValueError(f"{label}: cycle time must exceed zero.")
        if not 0 < stage.availability <= 1:
            raise ValueError(f"{label}: availability must be greater than 0% and at most 100%.")
        if not 0 < stage.yield_rate <= 1:
            raise ValueError(f"{label}: good-part yield must be greater than 0% and at most 100%.")
        if stage.run_power_kw <= 0 or stage.idle_power_kw < 0:
            raise ValueError(f"{label}: running power must be positive and idle power cannot be negative.")
        if stage.idle_power_kw > stage.run_power_kw * SPEED_MODES["Eco"][1]:
            raise ValueError(
                f"{label}: idle kW cannot exceed 78% of standard running kW, "
                "so that idle power does not exceed Eco-mode running power."
            )

    for name, value in (
        ("Shift hours", conditions.shift_hours),
        ("Demand", conditions.demand_units),
        ("Electricity tariff", conditions.electricity_tariff),
        ("Emission factor", conditions.emission_factor),
    ):
        _finite_number(value, name)
    if not 0 < conditions.shift_hours <= 24:
        raise ValueError("Shift hours must be greater than zero and no more than 24.")
    if conditions.demand_units < 0:
        raise ValueError("Demand cannot be negative.")
    if conditions.electricity_tariff < 0 or conditions.emission_factor < 0:
        raise ValueError("Tariff and emissions factor cannot be negative.")


def downstream_yields(stages: Sequence[Stage]) -> tuple[float, ...]:
    """Yield from each stage's input through the last stage (inclusive)."""
    running = 1.0
    reverse: list[float] = []
    for stage in reversed(stages):
        running *= stage.yield_rate
        reverse.append(running)
    return tuple(reversed(reverse))


def simulate_line(
    stages: Sequence[Stage],
    conditions: OperatingConditions,
    speed_modes: Sequence[str] | None = None,
) -> SimulationResult:
    """Return expected daily good output and in-shift electrical energy.

    Assumptions: a serial line, parallel identical machines at each stage,
    availability-adjusted operating hours, constant stage yield, and no
    work-in-process constraints. Machines are idle (not off) when available but
    not processing, and consume no energy when unavailable.
    """
    validate_inputs(stages, conditions)
    modes = tuple(speed_modes) if speed_modes is not None else ("Standard",) * len(stages)
    if len(modes) != len(stages):
        raise ValueError("Provide exactly one speed mode for each stage.")
    if any(mode not in SPEED_MODES for mode in modes):
        raise ValueError("Speed mode must be Eco, Standard, or Boost.")

    yields = downstream_yields(stages)
    capacity_inputs = []
    capacity_goods = []
    for stage, mode, downstream in zip(stages, modes, yields):
        cycle_multiplier, _ = SPEED_MODES[mode]
        input_capacity = stage.machines * 60 * conditions.shift_hours * stage.availability / (stage.cycle_time_min * cycle_multiplier)
        capacity_inputs.append(input_capacity)
        capacity_goods.append(input_capacity * downstream)

    bottleneck_idx = min(range(len(stages)), key=lambda i: capacity_goods[i])
    total_capacity = capacity_goods[bottleneck_idx]
    good_output = min(conditions.demand_units, total_capacity)
    metrics: list[StageMetrics] = []
    for stage, mode, downstream, input_capacity, good_capacity in zip(stages, modes, yields, capacity_inputs, capacity_goods):
        cycle_multiplier, power_multiplier = SPEED_MODES[mode]
        processed = good_output / downstream
        active_hours = processed * stage.cycle_time_min * cycle_multiplier / 60
        available_hours = stage.machines * conditions.shift_hours * stage.availability
        # All processing is within the available machine-hours by construction.
        active_hours = min(active_hours, available_hours)
        run_power = stage.run_power_kw * power_multiplier
        energy = active_hours * run_power + (available_hours - active_hours) * stage.idle_power_kw
        metrics.append(
            StageMetrics(
                name=stage.name,
                machines=stage.machines,
                speed_mode=mode,
                capacity_input_units=input_capacity,
                good_capacity_units=good_capacity,
                processed_input_units=processed,
                scrapped_units=processed * (1 - stage.yield_rate),
                active_machine_hours=active_hours,
                available_machine_hours=available_hours,
                utilization_pct=100 * active_hours / available_hours,
                energy_kwh=energy,
            )
        )
    total_energy = sum(stage.energy_kwh for stage in metrics)
    return SimulationResult(
        good_units=good_output,
        maximum_good_units=total_capacity,
        fulfillment_pct=(100 * good_output / conditions.demand_units) if conditions.demand_units > 0 else 100.0,
        total_energy_kwh=total_energy,
        emissions_kg=total_energy * conditions.emission_factor,
        electricity_cost=total_energy * conditions.electricity_tariff,
        kwh_per_good_unit=total_energy / good_output if good_output > 0 else None,
        bottleneck_stage=stages[bottleneck_idx].name,
        stage_metrics=tuple(metrics),
    )
