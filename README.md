# SustainFab

### Sustainable Manufacturing Systems Optimizer

A self-contained, interactive engineering project that models a **serial manufacturing line** and explores the trade-offs between **good-product throughput**, **electricity use**, **operational carbon emissions**, and **electricity cost**. Designed as an educational, transparent manufacturing-systems case study, with an auditable simulation model and exhaustive, bounded optimization.

**Stack:** Python · Streamlit · NumPy · Pandas · Plotly · Pytest

---

## Highlights

- **Calculate button:** edit operating conditions and machine-stage inputs first, then click **🧮 Calculate** to commit a validated scenario and refresh the results. Until clicked again, the dashboard continues displaying the previous calculated scenario with a warning if inputs change.
- **Editable manufacturing line:** up to six consecutive stages, each with 1–20 identical parallel machines, cycle time, availability, good-part yield, operating power, and idle power.
- **Line-wide production model:** calculates yield-adjusted capacity, bottleneck, stage input flow, expected scraps, machine utilization, and good-unit production.
- **Energy and carbon accounting:** daily operating + idle electricity; emissions from a configurable grid factor; electricity costs from a configurable tariff.
- **Discrete scenario optimizer:** enumerates all combinations of baseline ±1 machine per stage and three configurable operating-speed modes. The objective can be (1) **minimum electricity while meeting a desired production target**, or (2) **maximum good output within an electricity cap**.
- **Pareto visualization:** nondominated energy/throughput alternatives, with baseline and recommended design highlighted.
- **Demand sensitivity:** compare the original line and selected scenario over a demand sweep, without falsely claiming each point is reoptimized.
- **Shift-hour planning:** analytically estimate scheduled operating hours required to reach a good-product target and compare output/energy over 1–24 hours (not job-level sequencing).
- **Reproducible exports:** input CSV, performance comparison CSV, demand sensitivity CSV and complete results JSON.
- **Automated testing:** verification of throughput, stage yields, electrical energy, invalid inputs, infeasible constraints, and optimizer correctness against an independent brute-force benchmark.

## Currency conversion and reference rates

SustainFab keeps the **input electricity tariff in USD/kWh**, independent of
the cost display currency. Choosing USD ($), EUR (€), GBP (£), or NGN (₦)
converts the tariff and calculated electricity-only costs consistently; it
**does not** alter electricity consumption, production throughput, or CO₂ emissions.

The app fetches daily USD-based FX quotations from the
[ExchangeRate-API open-access endpoint](https://www.exchangerate-api.com/docs/free)
and caches them for 24 hours. It shows the provider and the last rate-update
date, and credits the service as required by its attribution policy.
If the service is unavailable or returns stale data, the dashboard shows a
visible **dated fallback**, using rates published by the West African Monetary
Agency on [16 November 2026](https://exrates.amao-wama.org/daily-rates?start=2026-11-16&end=2026-11-16&format=pdf):
1 USD = 0.904817 EUR, 0.755662 GBP, and 1,355.381500 NGN.
These fixed fallback values **are not live quotations** on subsequent dates.

The costs are illustrative USD-based electricity-cost assumptions converted
to the selected display currency, **not verified local electricity tariffs**.
For a local tariff, first convert it to its USD/kWh equivalent using the
currently shown reference rate. JSON exports include the tariff assumption,
rate table, FX source, currency and update date for auditability.

## Using the Calculate button

1. Choose the operating conditions in the sidebar (hours, demand, currency,
   USD reference electricity tariff, emissions factor), then edit the stage table.
2. Click the blue **🧮 Calculate** button directly beneath the manufacturing
   input table to validate and apply these settings.
3. Review the baseline, optimization, demand sensitivity, and operating-hours
   results. The **Optimize** tab's goal and constraint controls can still be
   varied interactively for the last calculated manufacturing inputs.
4. If you change sidebar settings or machine-stage inputs again, the dashboard
   **continues to display the last calculated scenario** and presents a warning
   rather than silently recalculating. Click **Calculate** again to apply changes.
5. Downloaded reports and the "last calculated line" CSV use the committed
   scenario, not unsaved input edits. The FX snapshot also stays consistent
   with the committed scenario until recalculation.

A fresh session displays the input editor first; calculations only appear after
you click **Calculate**. Invalid inputs result in an explicit error rather
than updating the saved scenario.

## 1. Launch locally

Prerequisites: Python 3.11 or 3.12.

```bash
python -m venv .venv
```

Activate the virtual environment:

- **Windows PowerShell:** `.venv\Scripts\Activate.ps1`
- **macOS / Linux:** `source .venv/bin/activate`

Then:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser, usually at `http://localhost:8501`.

### Run the validation tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## 2. Upload to GitHub

1. Create a **new, empty public repository** named **`SustainFab`** on GitHub; you do not need to add a GitHub-generated README.
2. Extract the distributed `SustainFab_GitHub_Ready.zip` file **on your computer**.
3. Open the extracted `SustainFab` folder, select its **contents** (not the ZIP itself), and drag them onto **Add file → Upload files** in the root of your new GitHub repository. Include `.streamlit` and `.github` directories.
4. Commit the files to the default branch.

**Important:** Uploading the ZIP file as a single file does *not* publish a usable Streamlit code repository. Upload the extracted files and directories. `app.py`, `requirements.txt`, and `README.md` must sit in the **repository root**.

### Deploy for free to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
2. Choose **Create app** (or **New app**) and select your `SustainFab` repository and the default branch.
3. Set the **main file path** to `app.py` and deploy.
4. Streamlit automatically reads `requirements.txt` to install dependencies.

The app is designed to run without API keys, database connections, uploaded secrets, or external data access.

## 3. Model inputs

| Field | Unit | Interpretation |
| --- | --- | --- |
| Stage | text | Unique name, in production flow order |
| Machines | count | Number of parallel identical machines at that stage |
| Cycle time | min/input unit | Time a machine spends processing one incoming unit |
| Availability | fraction / % | Fraction of scheduled time the stage's machines are operational |
| Good yield | fraction / % | Fraction of processed input units that pass the stage |
| Run power | kW/machine | Real power while processing, in Standard mode |
| Idle power | kW/machine | Power while available but not processing |
| Scheduled hours | h/day | Daily production window |
| Good-unit demand | units/day | Daily demand for finished **good** products |
| Electricity tariff | USD/kWh | User-provided reference assumption, converted to the selected display currency |
| Emission factor | kg CO₂/kWh | User-provided operational grid factor |

The included `data/example_line.csv` is a **fully synthetic illustration**; no figures claim to describe a real industrial facility. Replace all inputs with justified measurements for meaningful engineering interpretation.

## 4. Equations

Let stage *i* have `mᵢ` machines, availability `aᵢ`, cycle time `tᵢ` minutes per input unit, yield `yᵢ`, and daily scheduled hours `H`. The speed-mode cycle multiplier is `rᵢ`, with effective cycle time `tᵢ × rᵢ`.

**Stage input capacity, in incoming units/day:**

```text
C_input,i = m_i × (60 × H × a_i) / (t_i × r_i)
```

**Equivalent final-good capacity** contributed by the stage (all subsequent losses included):

```text
Y_downstream,i = product(y_j for j = i through last stage)
C_good,i = C_input,i × Y_downstream,i
```

**Expected finished good units/day:**

```text
Q_good = min(good_unit_demand, C_good,1, ..., C_good,N)
```

**Input units required at stage i to support the finished output:**

```text
Q_input,i = Q_good / Y_downstream,i
Expected_scrap_i = Q_input,i × (1 - y_i)
```

**Available and active machine-hours/day per stage:**

```text
T_available,i = m_i × H × a_i
T_active,i = Q_input,i × (t_i × r_i) / 60
Utilization_i = 100 × T_active,i / T_available,i
```

**Daily stage electricity:**

```text
E_i = T_active,i × (P_run,i × p_i) + (T_available,i - T_active,i) × P_idle,i
E_total = sum(E_i)
Cost = E_total × electricity_tariff
CO2_kg = E_total × grid_emission_factor
Energy_intensity = E_total / Q_good  [when Q_good > 0]
```

Here `pᵢ` is the speed-mode running-power multiplier. Machines are **assumed electrically off during unavailable time**, but draw idle power during available nonprocessing hours. Yield losses are accounted for stage by stage; the result is an **expected-value** model and can include fractional output units.

### Speed modes

| Mode | Cycle-time multiplier | Running-power multiplier | Interpretation |
| --- | ---: | ---: | --- |
| Eco | ×1.18 | ×0.78 | Slower, lower running power |
| Standard | ×1.00 | ×1.00 | Reference assumptions |
| Boost | ×0.82 | ×1.32 | Faster, higher running power |

These multipliers are **hypothetical scenario parameters** for comparison, **not measured equipment characteristics**. The app rejects inputs whose idle power would be higher than Eco-mode running power.

## 5. Optimization method

The optimizer uses deterministic **exhaustive enumeration**, not an unverified heuristic. For each stage it checks integer machine counts within `[max(1, baseline − 1), min(20, baseline + 1)]` and every speed mode. Thus, in the worst case of six stages with three allowed machine counts each, it checks `9^6 = 531,441` joint configurations, below its 600,000-case ceiling.

Every candidate uses the **same capacity, flow, and electricity equations** as the baseline. For the chosen result, the optimizer runs the independent scalar simulation a second time and checks that output and energy agree within numerical tolerance.

1. **Energy minimization:** choose minimum-kWh scenario with good output at least `target_fraction × good_unit_demand`, breaking energy ties in favor of higher output.
2. **Output maximization:** choose the maximum-good-output scenario below or at a specified kWh/day budget, breaking output ties in favor of lower energy.
3. If the constraint cannot be satisfied, the app **explicitly reports infeasibility** and labels the presented fallback as a best-effort scenario, never a feasible optimum.
4. The Pareto frontier retains only nondominated throughput-vs-energy outcomes. If more than 240 points occur, **only the chart** is downsampled; the solution always uses the full candidate set.

**What is optimized:** a bounded set of operating scenarios. **What is not optimized:** real-time maintenance, machines' purchase cost, safety constraints, sequencing or delivery due dates, layout, quality-control investments, production lot sizes, or other real-plant decisions.

### Shift-hour planning

For a fixed machine configuration and speed mode, the minimum required scheduled hours are calculated from the slowest stage's **yield-adjusted hourly final-good capacity**:

```text
R_good,i = m_i × 60 × a_i × Y_downstream,i / (t_i × r_i)
H_needed = target_good_units / min(R_good,i)
```

The app flags any resulting schedule exceeding 24 hours/day as infeasible for one day. This is **operating-hours planning**, not a job-shop scheduler or a guarantee of production in practice.

## 6. Project structure

```text
SustainFab/
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Runtime dependencies
├── requirements-dev.txt       # Testing dependencies
├── README.md                  # Setup, equations and deployment
├── LICENSE                    # MIT license
├── .gitignore
├── .streamlit/
│   └── config.toml            # App theme
├── .github/
│   └── workflows/
│       └── tests.yml          # Automated test checks on GitHub
├── data/
│   └── example_line.csv       # Synthetic input dataset
├── docs/
│   ├── VALIDATION.md          # Verification assumptions and tests
│   └── EXAMPLE_RESULTS.md     # Demonstrated sample scenario
├── sustainfab/
│   ├── __init__.py
│   ├── model.py                # Pure simulation engine
│   ├── optimizer.py            # Bounded exhaustive optimizer
│   └── planning.py             # Analytical shift-hour planning
└── tests/
    ├── test_model.py
    ├── test_optimizer.py
    └── test_planning.py
```

## 7. Limitations and suitable use

This is a **deterministic steady-state screening tool**, not a discrete-event production scheduler. It assumes an ordered serial line; identical parallel machines at each stage; homogeneous parts; constant cycle times, availability, yields and power; no queues, blocking, starvation transients, rework, buffers, setup/changeovers, demand timing, or partial machine shifts. Consequently, it does not establish real-world capacity guarantees or predictive reliability.

Carbon calculations represent **operational purchased-electricity emissions only**. They omit material embodied carbon, transport, thermal fuels, water, maintenance and equipment lifecycle impacts. Cost reports represent **electricity only** and do not include capex or labor. A speed-mode change is a numerical scenario, not proof that a machine can physically operate at that setting. Before actual deployment, recalibrate inputs with plant measurements, verify operating modes with manufacturers, and review feasibility and safety constraints.

See `docs/VALIDATION.md` for the unit tests and independent optimizer benchmark.

## License

MIT — see `LICENSE`.
