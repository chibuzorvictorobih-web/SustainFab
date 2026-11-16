"""Pure techno-economic, material-flow and uncertainty calculations.

All default engineering inputs are illustrations, not factory observations.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from math import isfinite
from typing import Sequence
import numpy as np
from .model import OperatingConditions, SimulationResult, Stage, simulate_line, validate_inputs

@dataclass(frozen=True)
class Economics:
    added_machines: int
    investment_usd: float
    annual_output_delta: float
    annual_electricity_delta_usd: float
    annual_maintenance_delta_usd: float
    annual_net_benefit_usd: float
    npv_usd: float
    simple_payback_years: float | None

def evaluate_economics(
    baseline: SimulationResult, selected: SimulationResult,
    baseline_stages: Sequence[Stage], selected_stages: Sequence[Stage], *,
    capital_per_machine_usd: float, installation_fraction: float,
    annual_maintenance_per_machine_usd: float, margin_per_good_unit_usd: float,
    electricity_tariff_usd_kwh: float, operating_days: int,
    analysis_years: int, discount_rate: float,
) -> Economics:
    """Incremental cash flow against existing machinery, without assumed salvage.

    Contribution margin is AFTER non-energy variable costs but BEFORE electricity
    and the maintenance costs entered here; do not count them twice.
    """
    vals = (capital_per_machine_usd, installation_fraction,
            annual_maintenance_per_machine_usd, margin_per_good_unit_usd,
            electricity_tariff_usd_kwh, discount_rate)
    if any(not isfinite(x) or x < 0 for x in vals):
        raise ValueError("Economic assumptions must be finite and nonnegative.")
    if installation_fraction > 3 or discount_rate > 1:
        raise ValueError("Installation fraction must be <=3 and discount rate <=1.")
    if isinstance(operating_days, bool) or not isinstance(operating_days, int) or not 1 <= operating_days <= 366:
        raise ValueError("Operating days must be an integer from 1 to 366.")
    if isinstance(analysis_years, bool) or not isinstance(analysis_years, int) or not 1 <= analysis_years <= 30:
        raise ValueError("Analysis years must be an integer from 1 to 30.")
    if len(baseline_stages) != len(selected_stages) or not baseline_stages:
        raise ValueError("The baseline and selected line must share a stage structure.")
    for base, new in zip(baseline_stages, selected_stages):
        if base.name != new.name:
            raise ValueError("Stage identities must match.")
    added = sum(max(0, s.machines - b.machines) for b, s in zip(baseline_stages, selected_stages))
    net_change = sum(s.machines - b.machines for b, s in zip(baseline_stages, selected_stages))
    capex = added * capital_per_machine_usd * (1 + installation_fraction)
    output_delta = (selected.good_units - baseline.good_units) * operating_days
    electricity_delta = (selected.total_energy_kwh - baseline.total_energy_kwh) * electricity_tariff_usd_kwh * operating_days
    maintenance_delta = net_change * annual_maintenance_per_machine_usd
    annual_benefit = output_delta * margin_per_good_unit_usd - electricity_delta - maintenance_delta
    npv = -capex + sum(annual_benefit / (1 + discount_rate) ** y for y in range(1, analysis_years + 1))
    payback = capex / annual_benefit if capex > 0 and annual_benefit > 0 else None
    return Economics(added, capex, output_delta, electricity_delta,
                     maintenance_delta, annual_benefit, npv, payback)

@dataclass(frozen=True)
class MaterialAssumptions:
    mass_per_input_unit_kg: float = 1.0
    embodied_kgco2e_per_kg: float = 2.0
    recovered_scrap_fraction: float = .5
    recovery_credit_kgco2e_per_kg: float = .5

@dataclass(frozen=True)
class MaterialImpact:
    input_material_kg: float
    scrap_kg: float
    recovered_kg: float
    unrecovered_kg: float
    gross_embodied_kgco2e: float
    potential_recovery_credit_kgco2e: float
    net_material_kgco2e: float
    combined_operational_and_material_kgco2e: float

def assess_materials(result: SimulationResult, assumptions: MaterialAssumptions) -> MaterialImpact:
    """Constant mass per part through serial stages; hypothetical recovery credit."""
    a = assumptions
    vals = (a.mass_per_input_unit_kg, a.embodied_kgco2e_per_kg,
            a.recovered_scrap_fraction, a.recovery_credit_kgco2e_per_kg)
    if any(not isfinite(x) or x < 0 for x in vals):
        raise ValueError("Material assumptions must be finite and nonnegative.")
    if a.recovered_scrap_fraction > 1 or a.recovery_credit_kgco2e_per_kg > a.embodied_kgco2e_per_kg:
        raise ValueError("Recovery cannot exceed 100%; credit cannot exceed virgin emissions factor.")
    if not result.stage_metrics:
        raise ValueError("The simulation must contain stages.")
    feed_mass = result.stage_metrics[0].processed_input_units * a.mass_per_input_unit_kg
    scrap_mass = sum(s.scrapped_units for s in result.stage_metrics) * a.mass_per_input_unit_kg
    recovered = scrap_mass * a.recovered_scrap_fraction
    gross = feed_mass * a.embodied_kgco2e_per_kg
    credit = recovered * a.recovery_credit_kgco2e_per_kg
    net = max(0., gross - credit)
    return MaterialImpact(feed_mass, scrap_mass, recovered, scrap_mass - recovered,
                          gross, credit, net, result.emissions_kg + net)

@dataclass(frozen=True)
class UncertaintyResult:
    samples: int
    seed: int
    target_good_units: float
    mean_good_units: float
    p05_good_units: float
    median_good_units: float
    p95_good_units: float
    target_meeting_probability: float
    mean_energy_kwh: float
    good_units_samples: tuple[float, ...]

def monte_carlo_throughput(
    stages: Sequence[Stage], conditions: OperatingConditions,
    speed_modes: Sequence[str] | None = None, *, samples: int = 500,
    seed: int = 42, availability_sd_pp: float = 3.,
    cycle_time_sd_pct: float = 5., yield_sd_pp: float = 1.,
    target_good_units: float | None = None,
) -> UncertaintyResult:
    """Clipped independent normal DAILY parameter shocks, not breakdown events.

    The probability estimate is strictly conditional on user-supplied variability
    and must not be characterized as a calibrated plant reliability forecast.
    """
    validate_inputs(stages, conditions)
    if isinstance(samples, bool) or not isinstance(samples, int) or not 50 <= samples <= 5000:
        raise ValueError("Samples must be an integer from 50 to 5000.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Seed must be a nonnegative 32-bit integer.")
    widths = (availability_sd_pp, cycle_time_sd_pct, yield_sd_pp)
    if any(not isfinite(x) or not 0 <= x <= 30 for x in widths):
        raise ValueError("Variability inputs must be percentages from 0 to 30.")
    target = conditions.demand_units if target_good_units is None else target_good_units
    if not isfinite(target) or target < 0:
        raise ValueError("Target must be finite and nonnegative.")
    rng = np.random.default_rng(seed)
    outputs, energies = [], []
    for _ in range(samples):
        varied = tuple(replace(
            s,
            availability=float(np.clip(rng.normal(s.availability, availability_sd_pp / 100), 1e-5, 1)),
            cycle_time_min=float(max(1e-5, rng.normal(s.cycle_time_min, s.cycle_time_min * cycle_time_sd_pct / 100))),
            yield_rate=float(np.clip(rng.normal(s.yield_rate, yield_sd_pp / 100), 1e-5, 1)),
        ) for s in stages)
        scenario = simulate_line(varied, conditions, speed_modes)
        outputs.append(scenario.good_units)
        energies.append(scenario.total_energy_kwh)
    arr = np.asarray(outputs, dtype=float)
    return UncertaintyResult(
        samples, seed, float(target), float(arr.mean()),
        float(np.percentile(arr, 5)), float(np.median(arr)),
        float(np.percentile(arr, 95)), float(np.mean(arr >= target - 1e-9)),
        float(np.mean(energies)), tuple(outputs),
    )
