"""Tests for capacity, yield propagation, electricity, and validation."""
from dataclasses import replace

import pytest

from sustainfab.model import OperatingConditions, Stage, simulate_line


def conditions(**changes):
    return replace(OperatingConditions(), **changes)


def test_one_stage_hand_calculation():
    stage = Stage("Machine", 2, 2.0, 0.75, 0.90, 6.0, 1.0)
    sim = simulate_line([stage], conditions(demand_units=100, emission_factor=0.5, electricity_tariff=0.2))
    processed = 100 / 0.90
    active = processed * 2 / 60
    expected_energy = active * 6 + (2 * 8 * 0.75 - active) * 1
    assert sim.maximum_good_units == pytest.approx(324)
    assert sim.good_units == pytest.approx(100)
    assert sim.stage_metrics[0].scrapped_units == pytest.approx(processed * 0.1)
    assert sim.total_energy_kwh == pytest.approx(expected_energy)
    assert sim.emissions_kg == pytest.approx(expected_energy * 0.5)
    assert sim.electricity_cost == pytest.approx(expected_energy * 0.2)
    assert sim.fulfillment_pct == pytest.approx(100)


def test_two_stage_yield_adjusted_bottleneck_and_flow():
    stages = [
        Stage("Slow first", 1, 3, 1, 0.8, 5, 1),
        Stage("Fast last", 2, 1, 1, 0.9, 3, 0.5),
    ]
    sim = simulate_line(stages, conditions(demand_units=1000))
    # First-stage input capacity = 480/3 = 160. Final-good equivalents 160*.8*.9.
    assert sim.good_units == pytest.approx(115.2)
    assert sim.bottleneck_stage == "Slow first"
    first, second = sim.stage_metrics
    assert first.processed_input_units == pytest.approx(160)
    assert second.processed_input_units == pytest.approx(128)
    assert second.processed_input_units == pytest.approx(first.processed_input_units * 0.8)
    assert first.utilization_pct == pytest.approx(100)
    assert second.utilization_pct < 100


def test_demand_zero_still_draws_idle_power():
    stage = Stage("A", 1, 2.0, 0.5, 0.9, 5, 1)
    sim = simulate_line([stage], conditions(demand_units=0))
    assert sim.good_units == 0
    assert sim.total_energy_kwh == pytest.approx(4.0)
    assert sim.kwh_per_good_unit is None
    assert sim.fulfillment_pct == 100


def test_speed_modes_have_expected_capacity_and_power():
    stage = Stage("A", 1, 2.0, 1.0, 1.0, 10, 1)
    op = conditions(demand_units=1000)
    eco = simulate_line([stage], op, ["Eco"])
    standard = simulate_line([stage], op)
    boost = simulate_line([stage], op, ["Boost"])
    assert eco.good_units < standard.good_units < boost.good_units
    assert standard.maximum_good_units == pytest.approx(240)
    assert eco.maximum_good_units == pytest.approx(240 / 1.18)
    assert boost.maximum_good_units == pytest.approx(240 / 0.82)
    assert standard.total_energy_kwh == pytest.approx(80)


@pytest.mark.parametrize("stage", [
    Stage("A", 0, 2, 0.9, 0.99, 5, 1),
    Stage("A", 1, 0, 0.9, 0.99, 5, 1),
    Stage("A", 1, 2, 0, 0.99, 5, 1),
    Stage("A", 1, 2, 0.9, 0, 5, 1),
    Stage("A", 1, 2, 0.9, 0.99, 5, 4.5),
    Stage("", 1, 2, 0.9, 0.99, 5, 1),
])
def test_invalid_stage_inputs_are_rejected(stage):
    with pytest.raises(ValueError):
        simulate_line([stage], conditions())


def test_duplicate_stages_rejected():
    a = Stage("Same", 1, 2, 1, 1, 5, 1)
    with pytest.raises(ValueError, match="Duplicate"):
        simulate_line([a, replace(a, name="same")], conditions())


def test_invalid_modes_rejected():
    stage = Stage("A", 1, 2, 1, 1, 5, 1)
    with pytest.raises(ValueError, match="mode"):
        simulate_line([stage], conditions(), ["Turbo"])
