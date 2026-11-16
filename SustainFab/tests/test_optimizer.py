"""Independent brute-force tests of both optimization objectives."""
from dataclasses import replace
from itertools import product

import pytest

from sustainfab.model import OperatingConditions, SPEED_MODES, Stage, simulate_line
from sustainfab.optimizer import apply_choices, optimize_line

STAGES = (
    Stage("Cut", 2, 2.5, 0.9, 0.96, 7, 0.8),
    Stage("Finish", 1, 1.7, 0.94, 0.98, 5, 0.7),
)
OP = OperatingConditions(shift_hours=8, demand_units=260, electricity_tariff=0.2, emission_factor=0.4)


def independent_all_scenarios():
    count_sets = [[1, 2, 3], [1, 2]]
    for counts in product(*count_sets):
        config = tuple(replace(s, machines=n) for s, n in zip(STAGES, counts))
        for modes in product(SPEED_MODES.keys(), repeat=len(STAGES)):
            yield counts, modes, simulate_line(config, OP, modes)


def test_minimum_energy_solution_matches_brute_force():
    target = 0.9 * OP.demand_units
    possibilities = [s for _, _, s in independent_all_scenarios() if s.good_units >= target - 1e-7]
    assert possibilities
    optim = optimize_line(STAGES, OP, objective="min_energy_for_target", target_fraction=0.9)
    assert optim.feasible
    assert optim.evaluated_candidates == 54
    assert optim.simulation.good_units >= target - 1e-7
    assert optim.simulation.total_energy_kwh == pytest.approx(min(s.total_energy_kwh for s in possibilities))
    configured = apply_choices(STAGES, optim.choices)
    independent = simulate_line(configured, OP, [c.speed_mode for c in optim.choices])
    assert optim.simulation == independent


def test_maximum_output_with_energy_cap_matches_brute_force():
    scenarios = list(independent_all_scenarios())
    costs = sorted(s.total_energy_kwh for _, _, s in scenarios)
    budget = costs[len(costs) // 2]
    eligible = [s for _, _, s in scenarios if s.total_energy_kwh <= budget + 1e-7]
    optim = optimize_line(STAGES, OP, objective="max_output_for_budget", energy_budget_kwh=budget)
    assert optim.feasible
    assert optim.simulation.total_energy_kwh <= budget + 1e-7
    assert optim.simulation.good_units == pytest.approx(max(s.good_units for s in eligible))


def test_infeasible_production_target_is_explicit():
    target = OperatingConditions(shift_hours=1.0, demand_units=5000)
    optim = optimize_line(STAGES, target, target_fraction=1, station_delta=0)
    assert not optim.feasible
    assert optim.feasible_count == 0
    assert "infeasible" in optim.status_message.lower()
    assert optim.simulation.good_units < 5000


def test_infeasible_energy_cap_is_explicit():
    optim = optimize_line(STAGES, OP, objective="max_output_for_budget", energy_budget_kwh=0.0)
    assert not optim.feasible
    assert optim.feasible_count == 0
    assert optim.simulation.total_energy_kwh > 0


def test_pareto_frontier_increases_with_energy_and_output():
    optim = optimize_line(STAGES, OP, target_fraction=0.4)
    points = optim.pareto_frontier
    assert points
    assert all(b.energy_kwh >= a.energy_kwh for a, b in zip(points, points[1:]))
    assert all(b.good_units > a.good_units for a, b in zip(points, points[1:]))


def test_rejects_invalid_search():
    with pytest.raises(ValueError, match="Station delta"):
        optimize_line(STAGES, OP, station_delta=3)
    with pytest.raises(ValueError, match="fraction"):
        optimize_line(STAGES, OP, target_fraction=1.5)
    with pytest.raises(ValueError, match="budget"):
        optimize_line(STAGES, OP, objective="max_output_for_budget", energy_budget_kwh=-1)


def test_six_stage_full_search_space_matches_scalar_solution():
    """Exercise the maximum 9^6 candidate space, not just toy search spaces."""
    stages = tuple(Stage(f"S{i}", 2, 2.0 + 0.15 * i, 0.94, 0.98, 5.0, 0.5) for i in range(6))
    op = OperatingConditions(shift_hours=8, demand_units=300)
    optimized = optimize_line(stages, op, target_fraction=0.8)
    assert optimized.evaluated_candidates == 9 ** 6
    configured = apply_choices(stages, optimized.choices)
    check = simulate_line(configured, op, [choice.speed_mode for choice in optimized.choices])
    assert optimized.simulation.good_units == pytest.approx(check.good_units)
    assert optimized.simulation.total_energy_kwh == pytest.approx(check.total_energy_kwh)
