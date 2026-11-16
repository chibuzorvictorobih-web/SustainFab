"""Regression checks for investment, material flow, and stochastic production."""
from dataclasses import replace
import pytest
from sustainfab.advanced import (
    MaterialAssumptions, assess_materials, evaluate_economics,
    monte_carlo_throughput,
)
from sustainfab.model import Stage, OperatingConditions, simulate_line

def test_economic_calculation_matches_manual_cashflow():
    base = (Stage("Station", 1, 2, .9, .9, 5, .5),)
    altered = (replace(base[0], machines=2),)
    op = OperatingConditions(demand_units=350, electricity_tariff=.1)
    x = simulate_line(base, op)
    y = simulate_line(altered, op)
    report = evaluate_economics(
        x, y, base, altered, capital_per_machine_usd=1000,
        installation_fraction=.1, annual_maintenance_per_machine_usd=100,
        margin_per_good_unit_usd=2, electricity_tariff_usd_kwh=.1,
        operating_days=250, analysis_years=2, discount_rate=0,
    )
    assert report.added_machines == 1
    assert report.investment_usd == pytest.approx(1100)
    assert report.annual_net_benefit_usd == pytest.approx(
        report.annual_output_delta * 2 - report.annual_electricity_delta_usd - 100
    )
    assert report.npv_usd == pytest.approx(-1100 + 2 * report.annual_net_benefit_usd)
    assert report.simple_payback_years == pytest.approx(1100 / report.annual_net_benefit_usd)

def test_economics_rejects_bad_assumptions():
    stage = (Stage("Stage", 1, 2, .9, .9, 5, .5),)
    sim = simulate_line(stage, OperatingConditions())
    with pytest.raises(ValueError):
        evaluate_economics(
            sim, sim, stage, stage, capital_per_machine_usd=-1,
            installation_fraction=.1, annual_maintenance_per_machine_usd=100,
            margin_per_good_unit_usd=2, electricity_tariff_usd_kwh=.1,
            operating_days=250, analysis_years=5, discount_rate=.08
        )

def test_material_scrap_mass_and_emissions():
    stage = (Stage("Stage", 1, 2, .9, .9, 5, .5),)
    sim = simulate_line(stage, OperatingConditions(demand_units=100))
    data = assess_materials(sim, MaterialAssumptions(1, 3, .5, 1))
    assert data.input_material_kg == pytest.approx(100 / .9)
    assert data.scrap_kg == pytest.approx(100 / .9 * .1)
    assert data.unrecovered_kg == pytest.approx(data.scrap_kg / 2)
    assert data.potential_recovery_credit_kgco2e == pytest.approx(data.recovered_kg)
    assert data.combined_operational_and_material_kgco2e == pytest.approx(
        sim.emissions_kg + data.net_material_kgco2e
    )

def test_materials_reject_invalid_credit():
    stage = (Stage("Stage", 1, 2, .9, .9, 5, .5),)
    sim = simulate_line(stage, OperatingConditions())
    with pytest.raises(ValueError):
        assess_materials(sim, MaterialAssumptions(1, 2, .5, 3))

def test_monte_carlo_seed_and_zero_variation():
    stages = (Stage("Stage", 1, 2, .9, .9, 5, .5),)
    case = OperatingConditions(demand_units=400)
    fixed = monte_carlo_throughput(
        stages, case, samples=50,
        availability_sd_pp=0, yield_sd_pp=0, cycle_time_sd_pct=0
    )
    assert fixed.mean_good_units == pytest.approx(simulate_line(stages, case).good_units)
    a = monte_carlo_throughput(stages, case, samples=50, seed=123)
    b = monte_carlo_throughput(stages, case, samples=50, seed=123)
    assert a.good_units_samples == b.good_units_samples
    assert 0 <= a.target_meeting_probability <= 1

def test_bad_sampling_count_rejected():
    stage = (Stage("Stage", 1, 2, .9, .9, 5, .5),)
    with pytest.raises(ValueError):
        monte_carlo_throughput(stage, OperatingConditions(), samples=2)
