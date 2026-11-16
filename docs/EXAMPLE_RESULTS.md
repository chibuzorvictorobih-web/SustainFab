# Worked example — included synthetic line

All figures below are produced using `data/example_line.csv` and the default operating conditions: **8 hours/day, 400 demanded good units/day, 0.15 currency units/kWh, and 0.45 kg CO₂/kWh**. These are illustrative inputs and are **not refinery or factory observations**.

## Baseline (Standard speed, original station counts)

| Quantity | Value |
| --- | ---: |
| Expected good products | 343.98 units/day |
| Daily electricity | 402.43 kWh/day |
| Electricity-only cost | 60.36 currency units/day |
| Operational CO₂ | 181.09 kg/day |
| Limiting stage | Machining |

## Exhaustive energy minimization, targeting 95% of demand

The target is **380 final good units/day**. Among **4,374** unique configurations, **480** meet or exceed that target, and the lowest-electricity eligible configuration is:

| Stage | Original machines → selected machines | Selected operating mode |
| --- | --- | --- |
| Cutting | 2 → 3 | Eco |
| Machining | 3 → 4 | Eco |
| Assembly | 2 → 2 | Standard |
| Inspection | 1 → 1 | Eco |

| Quantity | Selected scenario |
| --- | ---: |
| Expected good products | 388.68 units/day |
| Daily electricity | 423.67 kWh/day |
| Electricity-only cost | 63.55 currency units/day |
| Operational CO₂ | 190.65 kg/day |

**Interpretation:** the higher-output scenario consumes **more total daily electricity and emits more CO₂**, even though its electricity intensity per good unit is lower than the baseline. The optimizer is not claiming unconditional energy savings; it minimizes energy **subject to meeting a production target** the baseline cannot reach.

## Maximum output under a budget equal to baseline electricity

Setting the cap to **402.43 kWh/day** produces a feasible scenario with about **368.64 good units/day**, consuming about **401.03 kWh/day**. This alternative can be useful when the available daily electricity is the dominant constraint.

## Shift-hour planning

To produce **400 good units/day** under the same steady-state assumptions, the baseline requires approximately **9.30 scheduled hours/day**, compared with approximately **8.23 scheduled hours/day** for the selected 95%-target configuration. These are capacity estimates, not guaranteed shift completion times.

The selected 8-hour configuration is feasible for the **380-unit** optimization target, not for all 400 units within eight hours.

## Reproducing the example

Run the Streamlit app without changing the example data. Open **Optimize**, leave its goal at **Minimize electricity for a production target** and the slider at **95%**. To try the other case, select **Maximize output within an electricity cap** and enter the baseline electricity value shown under **Production overview**.
