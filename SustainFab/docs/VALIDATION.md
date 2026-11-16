# Engineering validation and tests

## Model purpose and validity

SustainFab estimates average daily capacity and energy for an idealized **serial** line using expected values. It is not a discrete-event model and does not predict the probability of meeting a deadline. The included values are synthetic. This validation is **mathematical/software verification**, not experimental validation against an actual industrial production line.

## Mathematical checks

1. **Single-stage hand calculation**: when cycle time is 2 minutes, two machines operate for 8 hours at 75% availability and 90% yield, expected final-good capacity is `2 × 60 × 8 × 0.75 / 2 × 0.9 = 324 units/day`. For demand 100, required processing is `100/0.9` input units, and the active-plus-idle energy equation is independently checked.
2. **Two-stage flow reconciliation**: upstream quality losses propagate to downstream processing. At 80% followed by 90% yield, first-stage input capacity 160 yields `160 × 0.8 × 0.9 = 115.2` final good units; the second stage processes only `160 × 0.8 = 128` input units.
3. **Zero-demand behavior**: production is zero but available machines still consume idle power (as specified); energy intensity is undefined rather than a division-by-zero exception.
4. **Mode/capacity checks**: Eco decreases rate, Boost increases rate, and Standard matches the underlying cycle and power values.
5. **Validation guards**: machine counts, yields, availability, power, cycle time, stage names, and mode selection are checked, including impossible Eco running-power assumptions.

## Optimizer verification

- The optimizer computes all discrete scenarios using vectorized NumPy operations. It then re-evaluates the selected design with `simulate_line()`, an independently implemented scalar model, and verifies output and kWh to numerical tolerance.
- `tests/test_optimizer.py` independently uses Python `itertools.product` and the scalar simulation to enumerate a small two-stage line. Both the minimum-energy / required-throughput optimum and maximum-output / energy-budget optimum match the independent brute-force benchmarks.
- Both production-target infeasibility and energy-cap infeasibility are tested. Neither is presented as feasible.
- Pareto results are checked for monotonically increasing throughput as energy increases (nondomination when minimizing energy and maximizing output).
- A six-stage **531,441-case** stress test verifies the complete largest search space with ±1 machine choices and three speed modes.
- The shift-hour planner is checked against an independent run of the capacity model at the mathematically required scheduled hours; infeasible (>24h) and zero targets are checked.

## Reproduce tests

From the repository root, with packages installed:

```bash
python -m pytest -q
```

**21 automated tests passed** in the packaged verification run. This number may change if the tests are expanded. The included GitHub Actions workflow also runs the test suite when code is pushed or updated.

## Known limitations

- No statistical process-variation or uncertainty model; yields and uptime are treated as average fractions.
- No buffer capacity, queue lengths, blocking, starvation, lot sequencing, setup time or full discrete-event timing.
- No stochastic breakdown and repair model; an availability fraction scales scheduled machine-hours.
- No equipment capital expenditure or personnel costs, and no production changeover/maintenance investment penalty when adding machines.
- No rework loops; all rejected units are counted as expected scrap.
- Constant electricity tariff and emissions factor, no time-of-use tariff or renewable dispatch.
- All machines within a stage are treated as identical and have consistent running and idle power.
- Eco/Boost mode factors are synthetic; they must not be taken as physical speed–power curves.
- Fixed power assumptions mean electricity cost and grid CO₂ depend on the same energy total and are not independent objectives.

**Engineering use requires measurements, calibration, manufacturer checks, and sensitivity/uncertainty analysis.**
