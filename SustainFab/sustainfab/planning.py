"""Coarse shift-hour planning for expected daily good-product targets.

This calculates an operating-hours requirement; it is NOT job sequencing,
finite-capacity discrete-event scheduling, or a labor rostering model.
"""
from __future__ import annotations

from typing import Sequence

from .model import OperatingConditions, SPEED_MODES, Stage, downstream_yields, validate_inputs


def required_shift_hours(
    stages: Sequence[Stage],
    conditions: OperatingConditions,
    target_good_units: float,
    speed_modes: Sequence[str] | None = None,
) -> float:
    """Return analytical minimum planned shift hours for a given output target.

    The hourly final-good capacity is the minimum of the yield-adjusted stage
    capacities per scheduled hour. This function returns hours even if the
    required value exceeds 24 (an infeasibility warning for a one-day window).
    """
    validate_inputs(stages, conditions)
    if not 0 <= target_good_units < float("inf"):
        raise ValueError("Planning target must be finite and nonnegative.")
    modes = tuple(speed_modes) if speed_modes is not None else ("Standard",) * len(stages)
    if len(modes) != len(stages) or any(m not in SPEED_MODES for m in modes):
        raise ValueError("Provide a valid Eco, Standard or Boost mode for every stage.")
    if target_good_units == 0:
        return 0.0
    downstream = downstream_yields(stages)
    good_per_hour = min(
        stage.machines * 60 * stage.availability * yields / (stage.cycle_time_min * SPEED_MODES[mode][0])
        for stage, mode, yields in zip(stages, modes, downstream)
    )
    return target_good_units / good_per_hour
