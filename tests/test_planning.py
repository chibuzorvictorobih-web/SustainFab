"""Checks analytical shift-hours calculation against actual line throughput."""
from dataclasses import replace

import pytest

from sustainfab.model import OperatingConditions, Stage, simulate_line
from sustainfab.planning import required_shift_hours


def test_exact_hours_for_two_stage_target():
    stages = [Stage("A", 1, 2, 1, 0.9, 4, 0.5), Stage("B", 2, 1, 1, 0.8, 4, 0.5)]
    conditions = OperatingConditions(shift_hours=8, demand_units=120)
    # A: 30 input units per h × 0.9 × 0.8 = 21.6 final good units/h.
    hours = required_shift_hours(stages, conditions, 120)
    assert hours == pytest.approx(120 / 21.6)
    simulation = simulate_line(stages, replace(conditions, shift_hours=hours))
    assert simulation.good_units == pytest.approx(120)
    under = simulate_line(stages, replace(conditions, shift_hours=hours * 0.99))
    assert under.good_units < 120


def test_shift_planning_zero_and_infeasibility():
    stages = [Stage("A", 1, 10, 0.5, 1, 4, 0.5)]
    c = OperatingConditions()
    assert required_shift_hours(stages, c, 0) == 0
    assert required_shift_hours(stages, c, 500) > 24
    with pytest.raises(ValueError):
        required_shift_hours(stages, c, -10)
