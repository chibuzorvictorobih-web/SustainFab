"""Portable PDF generation smoke test."""
from dataclasses import replace
from sustainfab.model import Stage, OperatingConditions, simulate_line
from sustainfab.advanced import (
    MaterialAssumptions, assess_materials, evaluate_economics, monte_carlo_throughput,
)
from sustainfab.report import build_pdf_report

def test_pdf_case_report():
    original = (Stage("Cutting", 1, 3, .9, .95, 4, .3),)
    changed = (replace(original[0], machines=2),)
    operating = OperatingConditions(demand_units=120, electricity_tariff=.15)
    baseline, selected = simulate_line(original, operating), simulate_line(changed, operating)
    econ = evaluate_economics(
        baseline, selected, original, changed, capital_per_machine_usd=1000,
        installation_fraction=.1, annual_maintenance_per_machine_usd=100,
        margin_per_good_unit_usd=2, electricity_tariff_usd_kwh=.15,
        operating_days=250, analysis_years=5, discount_rate=.08,
    )
    material = MaterialAssumptions()
    pdf = build_pdf_report(
        baseline=baseline, selected=selected, economics=econ,
        baseline_material=assess_materials(baseline, material),
        selected_material=assess_materials(selected, material),
        material_assumptions=material,
        baseline_uncertainty=monte_carlo_throughput(original, operating, samples=50),
        selected_uncertainty=monte_carlo_throughput(changed, operating, samples=50),
        currency="NGN", usd_to_currency=1355.38, fx_updated="2026-11-16",
        usd_tariff=.15, demand=120, operating_hours=8,
        assumptions_description="Synthetic assumptions; not industrial measurements.",
    )
    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 2500
