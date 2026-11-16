"""SustainFab — interactive sustainable manufacturing systems optimizer.

Launch with: streamlit run app.py
"""
from __future__ import annotations

from dataclasses import asdict, replace
from io import BytesIO
import hashlib
import json
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sustainfab.model import OperatingConditions, Stage, simulate_line
from sustainfab.currency import (SYMBOLS, format_money, get_usd_exchange_rates, tariff_in_currency,
                                 usd_to_selected, selected_to_usd)
from sustainfab.planning import required_shift_hours
from sustainfab.advanced import (
    MaterialAssumptions, assess_materials, evaluate_economics,
    monte_carlo_throughput,
)
from sustainfab.report import build_pdf_report
from sustainfab.optimizer import StageChoice, apply_choices, optimize_line

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "data" / "example_line.csv"
COLUMNS = [
    "Stage", "Machines", "Cycle time (min/unit)", "Availability (%)",
    "Good yield (%)", "Run power (kW)", "Idle power (kW)",
]

st.set_page_config(page_title="SustainFab | Manufacturing Optimizer", page_icon="🏭", layout="wide")
st.markdown(
    """<style>
    .block-container {padding-top: 2rem; max-width: 1400px;}
    h1, h2, h3 {letter-spacing: -0.025em;}
    [data-testid="stMetric"] {background: #172432; border: 1px solid #283949;
        padding: 16px 18px; border-radius: 12px;}
    [data-testid="stMetricLabel"] {color: #b5c5d4;}
    .subtle {color:#9cb3c5; font-size: 0.94rem;}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def example_df() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


@st.cache_data(show_spinner=False)
def optimize_cached(stages, conditions, objective, target_fraction, energy_budget_kwh):
    return optimize_line(
        stages,
        conditions,
        objective=objective,
        target_fraction=target_fraction,
        energy_budget_kwh=energy_budget_kwh,
        station_delta=1,
    )


def parse_stages(table: pd.DataFrame) -> tuple[Stage, ...]:
    if table.empty:
        raise ValueError("Add at least one manufacturing stage.")
    result = []
    for row_num, (_, row) in enumerate(table.iterrows(), start=1):
        try:
            values = {name: row[name] for name in COLUMNS}
            if any(pd.isna(v) for v in values.values()):
                raise ValueError("fill in every column")
            machines = float(values["Machines"])
            if not machines.is_integer():
                raise ValueError("the number of machines must be a whole number")
            result.append(
                Stage(
                    name=str(values["Stage"]).strip(),
                    machines=int(machines),
                    cycle_time_min=float(values["Cycle time (min/unit)"]),
                    availability=float(values["Availability (%)"]) / 100.0,
                    yield_rate=float(values["Good yield (%)"]) / 100.0,
                    run_power_kw=float(values["Run power (kW)"]),
                    idle_power_kw=float(values["Idle power (kW)"]),
                )
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Row {row_num}: {exc}") from exc
    return tuple(result)


def stage_table(result) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Stage": m.name,
            "Machines": m.machines,
            "Speed": m.speed_mode,
            "Good-output capacity (units/day)": m.good_capacity_units,
            "Processed (input units/day)": m.processed_input_units,
            "Expected scrap (units/day)": m.scrapped_units,
            "Utilization (%)": m.utilization_pct,
            "Energy (kWh/day)": m.energy_kwh,
        }
        for m in result.stage_metrics
    ])


def summary_table(baseline, candidate, currency):
    return pd.DataFrame(
        {
            "Metric": ["Good units/day", "Demand fulfilled (%)", "Electricity (kWh/day)",
                       "Energy intensity (kWh/good unit)", "Electricity cost/day",
                       "CO₂ emissions (kg/day)"],
            "Baseline": [baseline.good_units, baseline.fulfillment_pct, baseline.total_energy_kwh,
                         baseline.kwh_per_good_unit, baseline.electricity_cost, baseline.emissions_kg],
            "Candidate": [candidate.good_units, candidate.fulfillment_pct, candidate.total_energy_kwh,
                          candidate.kwh_per_good_unit, candidate.electricity_cost, candidate.emissions_kg],
            "Unit/currency": ["units", "%", "kWh", "kWh/unit", currency, "kg CO₂"],
        }
    )


def metric_num(num: float | None, digits: int = 1) -> str:
    return f"{num:,.{digits}f}" if num is not None else "n/a"


def stage_energy_figure(*results) -> go.Figure:
    figure = go.Figure()
    for label, result in results:
        frame = stage_table(result)
        figure.add_bar(name=label, x=frame["Stage"], y=frame["Energy (kWh/day)"],
                       hovertemplate="%{x}<br>%{y:,.1f} kWh/day<extra>%{fullData.name}</extra>")
    figure.update_layout(barmode="group", yaxis_title="Electricity (kWh/day)",
                         xaxis_title=None, legend_title_text="Scenario", margin=dict(l=0, r=0, t=18, b=0))
    return figure


with st.sidebar:
    st.markdown("### Operating conditions")
    shift_hours = st.number_input("Scheduled hours per day", 0.5, 24.0, 8.0, 0.5,
                                  help="A single daily production window; the model is not a shift scheduler.")
    demand = st.number_input("Demand for good units/day", min_value=0, max_value=1_000_000,
                             value=400, step=10)
    currency = st.selectbox("Cost display currency", ["USD", "NGN", "EUR", "GBP"], index=0)
    # Input tariff stays in USD/kWh; selecting a currency changes only its
    # equivalent value, so no accidental relabeling of the electricity price.
    usd_tariff = st.number_input(
        "Reference electricity tariff (USD/kWh)", min_value=0.0,
        max_value=1_000_000.0, value=0.15, step=0.01, format="%.3f",
        help="Illustrative electricity price in USD/kWh; displayed costs are converted with the listed FX rate.",
    )
    @st.cache_data(ttl=86400, show_spinner=False)
    def cached_exchange_rates():
        return get_usd_exchange_rates()

    fx_rates = cached_exchange_rates()
    tariff = tariff_in_currency(float(usd_tariff), currency, fx_rates)
    st.caption(
        f"Preview converted tariff: {format_money(tariff, currency, 4)}/kWh · "
        f"1 USD = {format_money(fx_rates.per_usd[currency], currency, 4)}"
    )
    if fx_rates.fallback:
        st.warning(f"Offline FX fallback: dated {fx_rates.as_of}; not a live quote.", icon="⚠️")
    else:
        st.caption(f"Reference rates updated {fx_rates.as_of}")
    st.markdown(f"[FX source: {fx_rates.provider}]({fx_rates.source_url})")
    st.caption("Indicative reference rates; not a bank's exchange quote.")
    factor = st.number_input("Grid emissions factor (kg CO₂/kWh)", 0.0, 10.0, 0.45, 0.01,
                             help="Illustrative default. Replace with a justified local or supplier factor.")
    st.divider()
    st.markdown("**Sample data**")
    st.download_button("Download example CSV", SAMPLE_PATH.read_bytes(), "sustainfab_example_line.csv",
                       "text/csv", use_container_width=True)
    uploaded = st.file_uploader("Or load your production-line CSV", type="csv",
                                help="Use the same column names as the downloadable example.")

st.title("🏭 SustainFab")
st.markdown("**Sustainable Manufacturing Systems Optimizer**")
st.markdown(
    '<div class="subtle">A transparent, deterministic model of serial manufacturing lines. '
    'Explore throughput, quality losses, energy, emissions and explicitly bounded design choices.</div>',
    unsafe_allow_html=True,
)
st.divider()
st.subheader("01 · Define the manufacturing line")
if uploaded is not None:
    try:
        raw_table = pd.read_csv(BytesIO(uploaded.getvalue()))
        if set(raw_table.columns) != set(COLUMNS):
            raise ValueError("The CSV must contain exactly the seven columns used in the example file.")
        raw_table = raw_table[COLUMNS]
    except Exception as exc:
        st.error(f"Unable to load CSV: {exc}")
        st.stop()
else:
    raw_table = example_df()

st.caption("The included data are illustrative, not measurements from an actual factory. Edit any cell, or add/remove a stage (maximum six).")
editor_key = ("manufacturing_editor_" + hashlib.sha256(uploaded.getvalue()).hexdigest()[:12]) if uploaded is not None else "manufacturing_editor_sample"
line_table = st.data_editor(
    raw_table,
    key=editor_key,
    hide_index=True,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Stage": st.column_config.TextColumn("Stage", help="Unique name of processing station/stage", required=True),
        "Machines": st.column_config.NumberColumn("Machines", min_value=1, max_value=20, step=1, required=True),
        "Cycle time (min/unit)": st.column_config.NumberColumn("Cycle time (min/unit)", min_value=0.01, format="%.2f", required=True),
        "Availability (%)": st.column_config.NumberColumn("Availability (%)", min_value=0.01, max_value=100.0, format="%.2f", required=True),
        "Good yield (%)": st.column_config.NumberColumn("Good yield (%)", min_value=0.01, max_value=100.0, format="%.2f", required=True),
        "Run power (kW)": st.column_config.NumberColumn("Run power (kW)", min_value=0.01, format="%.2f", required=True),
        "Idle power (kW)": st.column_config.NumberColumn("Idle power (kW)", min_value=0.0, format="%.2f", required=True),
    },
)
st.caption("Availability = fraction of scheduled hours when machines are operable. Good yield = share of parts that pass a stage. Power values are per machine.")

# The input widgets are editable drafts. Only the Calculate button commits
# a complete, validated snapshot for the dashboard and optimizer.
# This prevents reruns from silently changing results midway through editing.
draft_signature = (
    line_table.to_json(orient="split"),
    float(shift_hours),
    float(demand),
    float(usd_tariff),
    float(factor),
    currency,
    fx_rates.as_of,
    tuple(sorted(fx_rates.per_usd.items())),
)

st.caption(
    "Edit the manufacturing line and operating conditions, then click "
    "**Calculate** to apply your inputs to every results tab."
)
calculate_clicked = st.button(
    "🧮 Calculate", type="primary", use_container_width=True,
    help="Validate the current inputs and refresh the baseline, optimization, sensitivity, and planning results.",
    key="sustainfab_calculate",
)

if calculate_clicked:
    try:
        calculated_stages = parse_stages(line_table)
        calculated_conditions = OperatingConditions(
            shift_hours=float(shift_hours),
            demand_units=float(demand),
            electricity_tariff=float(tariff),
            emission_factor=float(factor),
        )
        calculated_baseline = simulate_line(calculated_stages, calculated_conditions)
    except ValueError as exc:
        st.error(f"Please correct the model inputs: {exc}")
        st.stop()

    st.session_state["sustainfab_calculated_snapshot"] = {
        "signature": draft_signature,
        "stages": calculated_stages,
        "conditions": calculated_conditions,
        "baseline": calculated_baseline,
        "currency": currency,
        "usd_tariff": float(usd_tariff),
        "fx_rates": fx_rates,
        "line_table": line_table.copy(deep=True),
    }
    st.success("Calculation complete. Results below reflect your submitted inputs.")

snapshot = st.session_state.get("sustainfab_calculated_snapshot")
if snapshot is None:
    st.info("Enter or edit the production parameters above, then click **🧮 Calculate** to view results.")
    st.stop()

if snapshot["signature"] != draft_signature:
    st.warning(
        "You have uncalculated input changes. The results below still use your "
        "**last calculated settings**. Click **Calculate** to apply the new inputs."
    )

# All result tabs and downloadable reports use this same committed snapshot.
# Current edits remain in their widgets until the next Calculate click.
stages = snapshot["stages"]
conditions = snapshot["conditions"]
baseline = snapshot["baseline"]
currency = snapshot["currency"]
usd_tariff = snapshot["usd_tariff"]
fx_rates = snapshot["fx_rates"]
line_table = snapshot["line_table"]

overview, optimization, sensitivity, shift_planner, methodology, advanced_tab = st.tabs(
    ["📊 Production overview", "⚙️ Optimize", "📈 Sensitivity",
     "🗓️ Shift-hour planning", "📖 Method & downloads", "🔬 Advanced analyses"]
)

with overview:
    st.subheader("02 · Baseline performance")
    top1, top2, top3 = st.columns(3)
    top1.metric("Good output (units/day)", f"{baseline.good_units:,.1f}")
    top2.metric("Demand satisfied", f"{baseline.fulfillment_pct:.1f}%")
    top3.metric("Bottleneck stage", baseline.bottleneck_stage)
    bottom1, bottom2, bottom3 = st.columns(3)
    bottom1.metric("Electrical energy", f"{baseline.total_energy_kwh:,.1f} kWh/day")
    bottom2.metric(f"Electricity cost ({currency}/day)", format_money(baseline.electricity_cost, currency))
    bottom3.metric("Operational CO₂", f"{baseline.emissions_kg:,.1f} kg/day")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Stage-limited good-output capacity")
        info = stage_table(baseline)
        bottleneck_color = ["#f3a85d" if n == baseline.bottleneck_stage else "#11A59A"
                            for n in info["Stage"]]
        figure = go.Figure(go.Bar(x=info["Stage"], y=info["Good-output capacity (units/day)"],
                                  marker_color=bottleneck_color,
                                  hovertemplate="%{x}<br>%{y:,.1f} good units/day<extra></extra>"))
        figure.add_hline(y=conditions.demand_units, line_dash="dot", line_color="#A6B8CA",
                         annotation_text="Demand")
        figure.update_layout(yaxis_title="Equivalent final good units/day", xaxis_title=None,
                             margin=dict(l=0, r=0, t=18, b=0))
        st.plotly_chart(figure, use_container_width=True)
    with right:
        st.markdown("#### Daily energy by stage")
        st.plotly_chart(stage_energy_figure(("Baseline", baseline)), use_container_width=True)

    st.markdown("#### Stage calculations")
    st.dataframe(info, hide_index=True, use_container_width=True,
                 column_config={
                     "Good-output capacity (units/day)": st.column_config.NumberColumn(format="%.1f"),
                     "Processed (input units/day)": st.column_config.NumberColumn(format="%.1f"),
                     "Expected scrap (units/day)": st.column_config.NumberColumn(format="%.1f"),
                     "Utilization (%)": st.column_config.NumberColumn(format="%.1f"),
                     "Energy (kWh/day)": st.column_config.NumberColumn(format="%.1f"),
                 })
    st.caption("The bottleneck is the stage with the lowest downstream-yield-adjusted capacity, even when daily demand is below that capacity.")

with optimization:
    st.subheader("03 · Explore feasible operating scenarios")
    st.info("The optimizer exhaustively checks every combination of one fewer, the same, or one additional machine per stage (within 1–20), "
            "and Eco, Standard or Boost speed modes. It finds the best scenario **within this defined search space**, not a universal factory optimum.")
    goal = st.radio("Optimization goal", ["Minimize electricity for a production target", "Maximize output within an electricity cap"],
                    horizontal=True)
    objective = "min_energy_for_target" if goal.startswith("Minimize") else "max_output_for_budget"
    target_fraction = 0.95
    budget_kwh = None
    if objective == "min_energy_for_target":
        target_percentage = st.slider("Required demand fulfillment (%)", 0, 100, 95, 1)
        target_fraction = target_percentage / 100.0
        st.caption(f"Required good output: {conditions.demand_units * target_fraction:,.1f} units/day")
    else:
        budget_kwh = st.number_input(
            "Maximum daily electricity (kWh/day)", min_value=0.0, max_value=1_000_000_000.0,
            value=float(round(baseline.total_energy_kwh * 1.05, 2)), step=10.0,
        )

    try:
        with st.spinner("Evaluating configurations and Pareto frontier..."):
            optimum = optimize_cached(stages, conditions, objective, target_fraction, budget_kwh)
    except ValueError as exc:
        st.warning(f"Optimization unavailable: {exc}")
        st.stop()

    if optimum.feasible:
        st.success(optimum.status_message)
    else:
        st.warning(optimum.status_message)
    st.caption(f"{optimum.evaluated_candidates:,} unique scenarios evaluated · "
               f"{optimum.feasible_count:,} satisfy the selected constraint · "
               "every reported result rechecked with the stand-alone simulation engine")

    result = optimum.simulation
    a, b, c = st.columns(3)
    a.metric("Selected output (units/day)", f"{result.good_units:,.1f}",
             delta=f"{result.good_units - baseline.good_units:+,.1f} vs baseline")
    b.metric("Selected energy", f"{result.total_energy_kwh:,.1f} kWh/day",
             delta=f"{result.total_energy_kwh - baseline.total_energy_kwh:+,.1f} kWh",
             delta_color="inverse")
    c.metric("Selected emissions", f"{result.emissions_kg:,.1f} kg/day",
             delta=f"{result.emissions_kg - baseline.emissions_kg:+,.1f} kg CO₂",
             delta_color="inverse")

    st.markdown("#### Scenario comparison")
    st.dataframe(summary_table(baseline, result, currency), hide_index=True, use_container_width=True,
                 column_config={"Baseline": st.column_config.NumberColumn(format="%.2f"),
                                "Candidate": st.column_config.NumberColumn(format="%.2f")})
    chosen = pd.DataFrame([
        {"Stage": s.name, "Baseline machines": s.machines,
         "Suggested machines": choice.machines, "Baseline mode": "Standard", "Suggested mode": choice.speed_mode,
         "Effective cycle (min/unit)": s.cycle_time_min * {"Eco": 1.18, "Standard": 1.0, "Boost": 0.82}[choice.speed_mode]}
        for s, choice in zip(stages, optimum.choices)
    ])
    st.dataframe(chosen, hide_index=True, use_container_width=True)

    plot_left, plot_right = st.columns(2)
    with plot_left:
        st.markdown("#### Output–energy Pareto frontier")
        frontier = optimum.pareto_frontier
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[p.good_units for p in frontier],
                                 y=[p.energy_kwh for p in frontier],
                                 name="Nondominated scenarios", mode="lines+markers",
                                 marker=dict(size=5, color="#11A59A"), line=dict(color="#11A59A"),
                                 hovertemplate="%{x:,.1f} units/day<br>%{y:,.1f} kWh/day<extra></extra>"))
        fig.add_trace(go.Scatter(x=[baseline.good_units], y=[baseline.total_energy_kwh],
                                 name="Baseline", mode="markers", marker=dict(size=14, color="#f3a85d", symbol="diamond")))
        fig.add_trace(go.Scatter(x=[result.good_units], y=[result.total_energy_kwh],
                                 name="Selected", mode="markers", marker=dict(size=14, color="#e4e9f1", symbol="star")))
        fig.update_layout(xaxis_title="Good units/day (higher is better)",
                          yaxis_title="kWh/day (lower is better)",
                          legend=dict(orientation="h", y=-0.25), margin=dict(l=0, r=0, t=18, b=40))
        st.plotly_chart(fig, use_container_width=True)
    with plot_right:
        st.markdown("#### Electricity by stage")
        st.plotly_chart(stage_energy_figure(("Baseline", baseline), ("Selected", result)),
                        use_container_width=True)
    st.caption("The frontier is a trade-off between good throughput and electrical energy. Cost and CO₂ are derived from energy using constant user-supplied factors; they are not independent optimization dimensions.")

    payload = {
        "application": "SustainFab",
        "data_note": "Example inputs, if unchanged, are illustrative; replace before engineering use.",
        "objective": optimum.objective,
        "constraint_feasible": optimum.feasible,
        "objective_message": optimum.status_message,
        "number_of_candidates": optimum.evaluated_candidates,
        "operating_conditions": asdict(conditions),
        "currency": currency,
        "electricity_tariff_reference": {"amount": float(usd_tariff), "currency": "USD", "unit": "kWh"},
        "exchange_rates": {
            "base": "USD", "per_usd": fx_rates.per_usd,
            "as_of": fx_rates.as_of, "provider": fx_rates.provider,
            "source_url": fx_rates.source_url, "fallback": fx_rates.fallback,
        },
        "electricity_cost_display_currency": currency,
        "input_stages": [asdict(s) for s in stages],
        "selected_configuration": [asdict(s) for s in optimum.choices],
        "baseline": asdict(baseline),
        "selected": asdict(result),
    }
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("Download complete result (JSON)",
                           json.dumps(payload, indent=2), "sustainfab_optimization.json",
                           "application/json", use_container_width=True)
    with d2:
        st.download_button("Download comparison (CSV)",
                           summary_table(baseline, result, currency).to_csv(index=False),
                           "sustainfab_comparison.csv", "text/csv", use_container_width=True)

with sensitivity:
    st.subheader("04 · Demand sensitivity")
    st.markdown("Keep the current machines and modes fixed while varying demand. This is a **what-if sweep**, not a re-optimization at every demand point.")
    scenarios = [("Baseline", stages, ["Standard"] * len(stages))]
    if "optimum" in locals():
        scenarios.append(("Selected configuration", apply_choices(stages, optimum.choices),
                          [choice.speed_mode for choice in optimum.choices]))
    scale = np.linspace(0.25, 1.5, 26)
    sweep_records = []
    for label, stage_config, modes in scenarios:
        for multiplier in scale:
            case = replace(conditions, demand_units=float(conditions.demand_units * multiplier))
            sim = simulate_line(stage_config, case, modes)
            sweep_records.append({
                "Scenario": label, "Demand (good units/day)": case.demand_units,
                "Output (good units/day)": sim.good_units,
                "Energy (kWh/day)": sim.total_energy_kwh,
                "CO2 (kg/day)": sim.emissions_kg,
            })
    sweep = pd.DataFrame(sweep_records)
    s1, s2 = st.columns(2)
    for column, target, title, axis_label in (
        (s1, "Output (good units/day)", "Good output vs demand", "Good units/day"),
        (s2, "Energy (kWh/day)", "Electricity vs demand", "kWh/day"),
    ):
        with column:
            st.markdown(f"#### {title}")
            fig = go.Figure()
            for label, group in sweep.groupby("Scenario", sort=False):
                fig.add_trace(go.Scatter(x=group["Demand (good units/day)"], y=group[target],
                                         name=label, mode="lines", line=dict(width=3)))
            fig.update_layout(xaxis_title="Demand (good units/day)", yaxis_title=axis_label,
                              margin=dict(l=0, r=0, t=18, b=0))
            st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download sensitivity data (CSV)", sweep.to_csv(index=False),
                       "sustainfab_sensitivity.csv", "text/csv")

with shift_planner:
    st.subheader("05 · Daily shift-hour planning")
    st.markdown("Estimate scheduled operating hours required to achieve a planned **good-unit target**. This is a capacity-based operating-hours planner, **not** sequencing of individual orders or employee shift rostering.")
    plan_target = st.number_input("Planned finished-good units/day", min_value=0, max_value=1_000_000,
                                  value=int(conditions.demand_units), step=10)
    plans = [("Baseline", stages, ["Standard"] * len(stages))]
    if "optimum" in locals():
        plans.append(("Selected configuration", apply_choices(stages, optimum.choices),
                      [choice.speed_mode for choice in optimum.choices]))
    plan_cols = st.columns(len(plans))
    for col, (label, stage_config, modes) in zip(plan_cols, plans):
        needed = required_shift_hours(stage_config, conditions, float(plan_target), modes)
        with col:
            st.metric(f"{label}: needed hours", f"{needed:,.2f} h/day")
            if needed > 24 + 1e-9:
                st.warning("Exceeds 24 hours: infeasible within one day at these settings.")
            else:
                st.caption("Target is capacity-feasible within a 24-hour daily window.")
    plan_records = []
    for label, stage_config, modes in plans:
        for planned_hours in np.arange(1, 25, dtype=float):
            new_case = replace(conditions, shift_hours=float(planned_hours), demand_units=float(plan_target))
            measured = simulate_line(stage_config, new_case, modes)
            plan_records.append({"Scenario": label, "Scheduled hours/day": planned_hours,
                                 "Expected good units/day": measured.good_units,
                                 "Electricity (kWh/day)": measured.total_energy_kwh})
    plan_df = pd.DataFrame(plan_records)
    plan_l, plan_r = st.columns(2)
    for col, y_key, subtitle in (
        (plan_l, "Expected good units/day", "Good output vs scheduled hours"),
        (plan_r, "Electricity (kWh/day)", "Energy vs scheduled hours"),
    ):
        with col:
            st.markdown(f"#### {subtitle}")
            graph = go.Figure()
            for label, sub in plan_df.groupby("Scenario", sort=False):
                graph.add_trace(go.Scatter(x=sub["Scheduled hours/day"], y=sub[y_key],
                                           name=label, mode="lines+markers"))
            graph.add_vline(x=conditions.shift_hours, line_dash="dot", line_color="#A6B8CA")
            graph.update_layout(xaxis_title="Scheduled hours/day", yaxis_title=y_key,
                                margin=dict(l=0, r=0, t=18, b=0))
            st.plotly_chart(graph, use_container_width=True)
    st.download_button("Download shift-hour planning data (CSV)", plan_df.to_csv(index=False),
                       "sustainfab_shift_hours.csv", "text/csv")

with methodology:
    st.subheader("06 · Model definition and reproducibility")
    st.markdown(dedent(
        """
        **What is modeled.** A deterministic, steady-state serial production line with identical
        parallel machines per stage; stage-level availability, yield, process time, and running/idle power.
        Machine availability is the share of scheduled time when equipment can run. No startup, changeover,
        repair-duration distribution, queues, inventories, labor constraints, capital costs, or upstream
        utility loads are modeled.

        **Good-output capacity.** For stage *i*, its available input capacity is
        `machines × 60 × shift_hours × availability / effective_cycle_min`. Multiply by all yields
        from that stage through the last stage to express capacity as final-good-equivalent units.
        Overall output is the minimum of demand and all stage good-equivalent capacities.

        **Energy.** Stage processing machine-hours are `required_input_units × effective_cycle_min / 60`.
        Stage electricity is `processing_machine_hours × running_kW + (available_machine_hours − processing_machine_hours) × idle_kW`.
        Machines are assumed **off during unavailable time**, and **idling when available but not processing**.

        **Speed modes (illustrative).** Eco changes cycle time by ×1.18 and running power by ×0.78;
        Standard by ×1.00 and ×1.00; Boost by ×0.82 and ×1.32. No claims are made that these coefficients
        represent actual machines. Recalibrate before real-world decisions.

        **Costs and carbon.** Electricity cost = kWh × tariff converted from its USD/kWh
        reference to the selected display currency with a dated exchange rate.
        FX source details are included in the downloadable JSON.
        Operational CO₂ = kWh × grid factor.
        No embodied/material emissions or equipment acquisition costs are included.

        **Shift planning.** Required scheduled hours are the desired good units divided by the smallest downstream-yield-adjusted stage capacity per scheduled hour. This does not sequence orders.

        **Optimization.** The entire discrete space of machine counts within ±1 of baseline per stage
        and three speed modes is enumerated (up to 600,000 configurations). The lowest-energy scenario
        meeting a given production target, or the highest-output scenario meeting an energy cap, is
        selected. If impossible, an explicitly labeled best-effort result is shown.

        **Interpretation.** Units and energy are **expected values**, not integer production orders.
        This is a screening and teaching model—not a validated plant-level digital twin or a substitute
        for instrumentation and engineering review.
        """
    ))
    st.markdown("#### Export the edited input line")
    st.download_button("Download last calculated line (CSV)", line_table.to_csv(index=False),
                       "sustainfab_current_line.csv", "text/csv")
    st.caption("Reproducible tests, derivations, and application/deployment instructions are included in the GitHub repository.")

def investment_amount_input(
    description: str, identifier: str, default_usd: float,
    currency: str, rates, step_usd: float, limit_usd: float,
) -> tuple[float, float]:
    """Enter a monetary assumption in the display currency, store USD internally.

    Each currency/rate has its own widget identity. A canonical USD value is
    synchronized on changes, so switching currencies cannot silently relabel
    an old input or reset financial assumptions to arbitrary defaults.
    """
    base_key = f"sustainfab_investment_usd_{identifier}"
    active_key = f"sustainfab_investment_active_{identifier}"
    rate = float(rates.per_usd[currency])
    widget_key = f"sf_invest_{identifier}_{currency}_{rate:.10f}"
    if base_key not in st.session_state:
        st.session_state[base_key] = float(default_usd)
    if st.session_state.get(active_key) != widget_key:
        st.session_state[widget_key] = round(
            usd_to_selected(st.session_state[base_key], currency, rates), 2
        )
        st.session_state[active_key] = widget_key

    def update_usd_reference():
        st.session_state[base_key] = selected_to_usd(
            st.session_state[widget_key], currency, rates
        )

    value = st.number_input(
        f"{description} ({SYMBOLS[currency]} {currency})",
        min_value=0.0,
        max_value=float(limit_usd * rate),
        step=float(max(.01, round(step_usd * rate, 2))),
        format="%.2f",
        key=widget_key,
        on_change=update_usd_reference,
        help=f"Enter the amount in {currency}; calculation converts to USD using the displayed FX reference.",
    )
    # Converting the visible numeric value directly guarantees a matching
    # result even in the initial run before the first on_change callback.
    return selected_to_usd(value, currency, rates), float(value)


with advanced_tab:
    st.subheader("07 · Advanced sustainability and investment analyses")
    st.caption(
        "Educational scenario screening only. All costs, material factors and "
        "uncertainty ranges below are editable **illustrative assumptions**, not measured factory data."
    )
    if not optimum.feasible:
        st.warning(
            "The optimization constraint is infeasible. Advanced comparisons "
            "use the optimizer's labelled best-effort scenario, not a feasible recommendation."
        )

    configured = apply_choices(stages, optimum.choices)
    selected_modes = tuple(choice.speed_mode for choice in optimum.choices)
    econ_tab, material_tab, risk_tab, report_tab = st.tabs(
        ["💰 Investment", "♻️ Materials and carbon",
         "🎲 Uncertainty", "📄 Case-study report"]
    )

    with econ_tab:
        st.markdown("#### Incremental techno-economic comparison")
        st.caption(
            "Compare selected equipment against the existing baseline. Capital "
            "expense counts only additional machines; no salvage value is assumed. "
            "Contribution margin must exclude electricity and the maintenance below."
        )
        rate = float(fx_rates.per_usd[currency])
        st.caption(
            f"Investment inputs and results use {SYMBOLS[currency]} ({currency}). "
            f"Reference rate: 1 USD = {format_money(rate, currency, 4)} "
            f"(updated {fx_rates.as_of}). Amounts are converted to USD "
            "internally for consistent financial analysis."
        )
        e1, e2, e3 = st.columns(3)
        with e1:
            machine_capex, machine_capex_local = investment_amount_input(
                "Purchase cost per additional machine", "machine_capex",
                12000.0, currency, fx_rates, 500.0, 100000000.0
            )
            margin_per_unit, margin_per_unit_local = investment_amount_input(
                "Contribution margin per good unit", "contribution",
                5.0, currency, fx_rates, .5, 1000000.0
            )
        with e2:
            install_pct = st.number_input(
                "Installation cost (% of machine purchase price)", 0.0, 300.0,
                10.0, 1.0, key="adv_install"
            )
            annual_maint, annual_maint_local = investment_amount_input(
                "Maintenance cost per machine per year", "maintenance",
                500.0, currency, fx_rates, 50.0, 100000000.0
            )
        with e3:
            operating_days = st.number_input(
                "Operating days per year", 1, 366, 250, 1, key="adv_days"
            )
            horizon = st.number_input(
                "Investment horizon (years)", 1, 30, 5, 1, key="adv_horizon"
            )
            discount_pct = st.number_input(
                "Discount rate (%/year)", 0.0, 100.0, 8.0, 1.0,
                key="adv_discount"
            )
        economic = evaluate_economics(
            baseline, result, stages, configured,
            capital_per_machine_usd=machine_capex,
            installation_fraction=install_pct / 100,
            annual_maintenance_per_machine_usd=annual_maint,
            margin_per_good_unit_usd=margin_per_unit,
            electricity_tariff_usd_kwh=usd_tariff,
            operating_days=int(operating_days),
            analysis_years=int(horizon),
            discount_rate=discount_pct / 100,
        )
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Additional machine investment",
                   format_money(economic.investment_usd * rate, currency))
        ec2.metric("Annual net benefit",
                   format_money(economic.annual_net_benefit_usd * rate, currency))
        ec3.metric("Net present value",
                   format_money(economic.npv_usd * rate, currency))
        st.metric("Simple payback",
                  f"{economic.simple_payback_years:.2f} years"
                  if economic.simple_payback_years is not None else "Not applicable")
        st.dataframe(pd.DataFrame([
            {"Indicator": "Annual change in finished good units",
             "Value": economic.annual_output_delta, "Unit": "units/year"},
            {"Indicator": "Annual change in electricity expense",
             "Value": economic.annual_electricity_delta_usd * rate,
             "Unit": currency + "/year"},
            {"Indicator": "Annual change in equipment maintenance",
             "Value": economic.annual_maintenance_delta_usd * rate,
             "Unit": currency + "/year"},
        ]), hide_index=True, use_container_width=True)
        st.caption(
            "NPV discounts constant nominal annual cash flows. It excludes taxes, "
            "depreciation, financing, disposal proceeds, installation downtime and "
            "machine-specific procurement constraints. Positive NPV under assumed "
            "values does not establish a real investment case."
        )

    with material_tab:
        st.markdown("#### Material throughput, waste and indicative embodied emissions")
        m1, m2 = st.columns(2)
        with m1:
            material_mass = st.number_input(
                "Material mass per input part (kg)", 0.001, 1000000.0, 1.0, .1,
                key="adv_material_mass"
            )
            virgin_factor = st.number_input(
                "Virgin material emissions factor (kg CO₂e/kg)", 0.0, 1000.0,
                2.0, .1, key="adv_material_factor"
            )
        with m2:
            recovery_fraction = st.slider(
                "Share of scrap recovered (%)", 0, 100, 50, 1,
                key="adv_recovery_fraction"
            )
            recovery_credit = st.number_input(
                "Potential recovery credit (kg CO₂e/kg)", 0.0,
                float(virgin_factor), min(.5, float(virgin_factor)),
                .1, key="adv_recovery_credit"
            )
        material_assumptions = MaterialAssumptions(
            material_mass, virgin_factor, recovery_fraction / 100, recovery_credit
        )
        material_base = assess_materials(baseline, material_assumptions)
        material_selected = assess_materials(result, material_assumptions)
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Selected scrap", f"{material_selected.scrap_kg:,.1f} kg/day")
        mc2.metric("Unrecovered scrap",
                   f"{material_selected.unrecovered_kg:,.1f} kg/day")
        mc3.metric("Selected net material emissions",
                   f"{material_selected.net_material_kgco2e:,.1f} kg CO₂e/day")
        st.dataframe(pd.DataFrame([
            {
                "Scenario": label,
                "Input material (kg/day)": analysis.input_material_kg,
                "Scrap (kg/day)": analysis.scrap_kg,
                "Recovered (kg/day)": analysis.recovered_kg,
                "Gross embodied CO₂e (kg/day)": analysis.gross_embodied_kgco2e,
                "Potential recovery credit (kg/day CO₂e)": analysis.potential_recovery_credit_kgco2e,
                "Combined operational + net material (kg CO₂e/day)":
                    analysis.combined_operational_and_material_kgco2e,
            } for label, analysis in (
                ("Baseline", material_base), ("Selected", material_selected)
            )
        ]), hide_index=True, use_container_width=True)
        st.caption(
            "Constant part mass is assumed through all stages. This is a simplified "
            "material balance, not a full lifecycle assessment. Recovered material "
            "only earns the specified **hypothetical** displacement credit; no "
            "measured carbon savings are implied."
        )

    with risk_tab:
        st.markdown("#### Monte Carlo throughput uncertainty")
        st.caption(
            "Independent, clipped-normal daily variations in uptime, cycle time "
            "and stage yield; not an event-based breakdown/repair simulation. "
            "Probabilities are conditional on the assumptions you enter."
        )
        u1, u2, u3 = st.columns(3)
        with u1:
            draws = st.number_input(
                "Random scenarios", 50, 5000, 500, 50, key="adv_draws"
            )
            rng_seed = st.number_input(
                "Reproducibility seed", 0, 9999999, 42, 1, key="adv_seed"
            )
        with u2:
            availability_sd = st.slider(
                "Availability standard deviation (percentage points)", 0.0, 30.0,
                3.0, .5, key="adv_uptime_std"
            )
            cycle_sd = st.slider(
                "Cycle-time standard deviation (%)", 0.0, 30.0,
                5.0, .5, key="adv_cycle_std"
            )
        with u3:
            yield_sd = st.slider(
                "Yield standard deviation (percentage points)", 0.0, 30.0,
                1.0, .5, key="adv_yield_std"
            )
            if optimum.objective == "min_energy_for_target":
                initial_target = float(optimum.target_units)
            else:
                initial_target = float(conditions.demand_units)
            st.caption(f"Target threshold: {initial_target:,.1f} good units/day")

        @st.cache_data(show_spinner=False)
        def run_sample_cached(stages_in, conditions_in, modes_in, n, seed_in,
                              uptime_in, cycle_in, yield_in, target_in):
            return monte_carlo_throughput(
                stages_in, conditions_in, modes_in,
                samples=n, seed=seed_in,
                availability_sd_pp=uptime_in,
                cycle_time_sd_pct=cycle_in,
                yield_sd_pp=yield_in, target_good_units=target_in,
            )

        with st.spinner("Sampling daily production scenarios..."):
            risk_base = run_sample_cached(
                stages, conditions, tuple("Standard" for _ in stages),
                int(draws), int(rng_seed), availability_sd, cycle_sd,
                yield_sd, initial_target
            )
            risk_selected = run_sample_cached(
                configured, conditions, selected_modes, int(draws), int(rng_seed),
                availability_sd, cycle_sd, yield_sd, initial_target
            )
        rc1, rc2 = st.columns(2)
        rc1.metric("Baseline probability of target",
                   f"{risk_base.target_meeting_probability:.1%}")
        rc2.metric("Selected probability of target",
                   f"{risk_selected.target_meeting_probability:.1%}")
        st.dataframe(pd.DataFrame([
            {
                "Scenario": label,
                "Mean output (units/day)": analysis.mean_good_units,
                "5th percentile": analysis.p05_good_units,
                "Median": analysis.median_good_units,
                "95th percentile": analysis.p95_good_units,
                "Mean energy (kWh/day)": analysis.mean_energy_kwh,
            } for label, analysis in (
                ("Baseline", risk_base), ("Selected", risk_selected)
            )
        ]), hide_index=True, use_container_width=True)
        histogram = go.Figure()
        histogram.add_trace(go.Histogram(
            x=risk_base.good_units_samples, name="Baseline", opacity=.65))
        histogram.add_trace(go.Histogram(
            x=risk_selected.good_units_samples, name="Selected", opacity=.65))
        histogram.add_vline(x=initial_target, line_dash="dash", line_color="#f3a85d")
        histogram.update_layout(
            barmode="overlay", xaxis_title="Good units/day",
            yaxis_title="Number of simulations",
        )
        st.plotly_chart(histogram, use_container_width=True)
        st.caption(
            "The 5th–95th percentiles describe the simulated outcome spread, "
            "not a statistical confidence interval for a calibrated physical plant."
        )

    with report_tab:
        st.markdown("#### Downloadable engineering case study")
        st.write(
            "The PDF includes the deterministic comparison, simplified economic "
            "and material assessments, Monte Carlo summary, FX reference date, "
            "and modelling limitations."
        )
        assumptions_text = (
            f"Illustrative assumptions in {currency}: "
            f"{currency} {machine_capex_local:,.2f} per new machine; "
            f"{install_pct:.1f}% installation; "
            f"{currency} {annual_maint_local:,.2f}/machine/year maintenance; "
            f"{currency} {margin_per_unit_local:,.2f}/unit contribution margin; "
            f"{operating_days} days/year; {horizon} years; "
            f"{discount_pct:.1f}% discount rate."
        )
        pdf = build_pdf_report(
            baseline=baseline, selected=result, economics=economic,
            baseline_material=material_base,
            selected_material=material_selected,
            material_assumptions=material_assumptions,
            baseline_uncertainty=risk_base,
            selected_uncertainty=risk_selected,
            currency=currency, usd_to_currency=rate,
            fx_updated=fx_rates.as_of, usd_tariff=usd_tariff,
            demand=conditions.demand_units,
            operating_hours=conditions.shift_hours,
            assumptions_description=assumptions_text,
        )
        st.download_button(
            "📄 Download engineering case study (PDF)",
            data=pdf, file_name="SustainFab_engineering_case_study.pdf",
            mime="application/pdf", use_container_width=True,
        )
        advanced_export = {
            "note": "Illustrative assumed inputs; not calibrated industrial findings.",
            "baseline": asdict(baseline), "selected": asdict(result),
            "economic_assumptions": assumptions_text,
            "investment_input_currency": currency,
            "investment_inputs_display_currency": {
                "purchase_per_machine": machine_capex_local,
                "annual_maintenance_per_machine": annual_maint_local,
                "margin_per_good_unit": margin_per_unit_local,
            },
            "investment_inputs_usd": {
                "purchase_per_machine": machine_capex,
                "annual_maintenance_per_machine": annual_maint,
                "margin_per_good_unit": margin_per_unit,
            },
            "economic_results_usd": asdict(economic),
            "economic_results_display_currency": {
                "currency": currency,
                "investment": economic.investment_usd * rate,
                "annual_net_benefit": economic.annual_net_benefit_usd * rate,
                "npv": economic.npv_usd * rate,
                "annual_electricity_delta": economic.annual_electricity_delta_usd * rate,
                "annual_maintenance_delta": economic.annual_maintenance_delta_usd * rate,
            },
            "material_assumptions": asdict(material_assumptions),
            "baseline_material": asdict(material_base),
            "selected_material": asdict(material_selected),
            "uncertainty_assumptions": {
                "draws": int(draws), "seed": int(rng_seed),
                "availability_sd_pp": availability_sd,
                "cycle_time_sd_pct": cycle_sd, "yield_sd_pp": yield_sd,
                "target_good_units": initial_target,
            },
            "uncertainty_baseline": {
                k: v for k, v in asdict(risk_base).items()
                if k != "good_units_samples"
            },
            "uncertainty_selected": {
                k: v for k, v in asdict(risk_selected).items()
                if k != "good_units_samples"
            },
            "currency": currency,
            "fx_per_usd": fx_rates.per_usd,
            "fx_date": fx_rates.as_of,
            "fx_source": fx_rates.provider,
        }
        st.download_button(
            "Download advanced results (JSON)",
            data=json.dumps(advanced_export, indent=2),
            file_name="SustainFab_advanced_analysis.json",
            mime="application/json", use_container_width=True,
        )
