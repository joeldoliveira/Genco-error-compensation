# Methodology notes

## Active-power prediction error

For each scenario and bus, define the signed active-power prediction error as

`e_Pg = Pg_pred - Pg_target`.

The aggregate signed error across the selected generator buses is the sum of these errors for a scenario. The absolute value of this sum can be smaller than the sum of the absolute individual errors when errors have opposite signs. This is referred to here as *error compensation*; it is not, by itself, proof of a physical mechanism or a network power-balance result.

## Before final publication

- Verify the challenge's bus indexing convention and which buses represent generators.
- Verify whether non-generator bus entries are absent, masked, or exactly zero by definition.
- Confirm the official definition of MRES before using that name in the report.
- If apparent power is analysed, calculate it consistently from active and reactive power: `S = sqrt(P^2 + Q^2)`, and clearly distinguish target-derived and prediction-derived values.
- Report the number of valid scenarios and how missing or duplicate scenario/bus rows are handled.
