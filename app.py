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
from sustainfab.planning import required_shift_hours
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
    tariff = st.number_input(f"Electricity tariff ({currency}/kWh)", 0.0, 1_000_000.0, 0.15, 0.01,
                             help="User-supplied assumption; no actual electricity tariff is claimed.")
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

try:
    stages = parse_stages(line_table)
    conditions = OperatingConditions(
        shift_hours=float(shift_hours), demand_units=float(demand),
        electricity_tariff=float(tariff), emission_factor=float(factor),
    )
    baseline = simulate_line(stages, conditions)
except ValueError as exc:
    st.error(f"Please correct the model inputs: {exc}")
    st.stop()

overview, optimization, sensitivity, shift_planner, methodology = st.tabs(
    ["📊 Production overview", "⚙️ Optimize", "📈 Sensitivity", "🗓️ Shift-hour planning", "📖 Method & downloads"]
)

with overview:
    st.subheader("02 · Baseline performance")
    top1, top2, top3 = st.columns(3)
    top1.metric("Good output", f"{baseline.good_units:,.1f} units/day")
    top2.metric("Demand satisfied", f"{baseline.fulfillment_pct:.1f}%")
    top3.metric("Bottleneck stage", baseline.bottleneck_stage)
    bottom1, bottom2, bottom3 = st.columns(3)
    bottom1.metric("Electrical energy", f"{baseline.total_energy_kwh:,.1f} kWh/day")
    bottom2.metric("Electricity cost", f"{baseline.electricity_cost:,.2f} {currency}/day")
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
    a.metric("Selected good output", f"{result.good_units:,.1f} units/day",
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

        **Costs and carbon.** Electricity cost = kWh × tariff. Operational CO₂ = kWh × grid factor.
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
    st.download_button("Download current line (CSV)", line_table.to_csv(index=False),
                       "sustainfab_current_line.csv", "text/csv")
    st.caption("Reproducible tests, derivations, and application/deployment instructions are included in the GitHub repository.")
