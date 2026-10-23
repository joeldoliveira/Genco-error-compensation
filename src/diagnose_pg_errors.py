
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

SYSTEMS = {
    "IEEE-14": ROOT / "outputs" / "case14_seed42"
              / "case14_ieee_predictions.parquet",
    "IEEE-30": ROOT / "outputs" / "case30_seed42"
              / "case30_ieee_predictions.parquet",
}

OUTPUT_DIR = ROOT / "outputs" / "pg_error_diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOLERANCE = 1e-6


# ============================================================
# Diagnostic analysis
# ============================================================

def diagnose_system(system_name, file_path):

    print("\n" + "=" * 90)
    print(f"ACTIVE POWER ERROR DIAGNOSTICS — {system_name}")
    print("=" * 90)

    df = pd.read_parquet(file_path)

    # Calculate prediction error
    df["Pg_error"] = df["Pg_pred"] - df["Pg_target"]

    # Summarize prediction errors and target values by bus
    summary = df.groupby("bus").agg(
        n_scenarios=("scenario", "nunique"),
        mean_Pg_target_MW=("Pg_target", "mean"),
        std_Pg_target_MW=("Pg_target", "std"),
        min_Pg_target_MW=("Pg_target", "min"),
        max_Pg_target_MW=("Pg_target", "max"),
        mean_Pg_pred_MW=("Pg_pred", "mean"),
        std_Pg_pred_MW=("Pg_pred", "std"),
        MAE_MW=("Pg_error", lambda x: x.abs().mean()),
        RMSE_MW=("Pg_error", lambda x: np.sqrt(np.mean(x**2))),
        max_abs_error_MW=("Pg_error", lambda x: x.abs().max()),
        nonzero_errors=("Pg_error", lambda x: (x.abs() > TOLERANCE).sum()),
    )

    # Add bus-type indicators when available
    for flag in ["PQ", "PV", "REF"]:
        if flag in df.columns:
            flag_summary = df.groupby("bus")[flag].mean()
            summary[f"mean_{flag}_flag"] = flag_summary

    summary = summary.reset_index()

    # Display buses with the largest errors first
    summary = summary.sort_values("MAE_MW", ascending=False)

    print("\nBus-level active power error summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    summary_path = OUTPUT_DIR / (
        f"{system_name.lower().replace('-', '')}_bus_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    # --------------------------------------------------------
    # Check whether target and predicted values are identical
    # --------------------------------------------------------

    print("\nTarget and prediction consistency checks:")

    for bus in sorted(df["bus"].unique()):

        bus_data = df[df["bus"] == bus]

        target_unique = bus_data["Pg_target"].nunique()
        prediction_unique = bus_data["Pg_pred"].nunique()

        target_std = bus_data["Pg_target"].std()
        prediction_std = bus_data["Pg_pred"].std()

        error_count = (
            bus_data["Pg_error"].abs() > TOLERANCE
        ).sum()

        print(
            f"Bus {bus:>2}: "
            f"unique targets={target_unique:>5}, "
            f"unique predictions={prediction_unique:>5}, "
            f"target std={target_std:.6f}, "
            f"prediction std={prediction_std:.6f}, "
            f"nonzero errors={error_count:>5}"
        )

    # --------------------------------------------------------
    # Verify whether errors are concentrated at buses 0 and 1
    # --------------------------------------------------------

    error_by_bus = df.groupby("bus")["Pg_error"].apply(
        lambda x: x.abs().sum()
    )

    total_absolute_error = error_by_bus.sum()

    if total_absolute_error > 0:
        share_bus_0 = 100 * error_by_bus.get(0, 0) / total_absolute_error
        share_bus_1 = 100 * error_by_bus.get(1, 0) / total_absolute_error
    else:
        share_bus_0 = 0.0
        share_bus_1 = 0.0

    print("\nAbsolute error concentration:")
    print(f"Bus 0 share: {share_bus_0:.4f}%")
    print(f"Bus 1 share: {share_bus_1:.4f}%")
    print(f"Combined share: {share_bus_0 + share_bus_1:.4f}%")

    print(f"\nBus summary saved to: {summary_path}")


# ============================================================
# Run diagnostics
# ============================================================

for system_name, file_path in SYSTEMS.items():

    if not file_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {file_path}"
        )

    diagnose_system(system_name, file_path)


print("\n" + "=" * 90)
print("DIAGNOSTIC ANALYSIS COMPLETED")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 90)
