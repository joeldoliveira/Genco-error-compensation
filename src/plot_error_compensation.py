
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

# Project root directory
ROOT = Path(__file__).resolve().parent.parent

# Input prediction files
SYSTEMS = {
    "IEEE-14": (
        ROOT
        / "outputs"
        / "case14_seed42"
        / "case14_ieee_predictions.parquet"
    ),
    "IEEE-30": (
        ROOT
        / "outputs"
        / "case30_seed42"
        / "case30_ieee_predictions.parquet"
    ),
}

# Output directory for figures
OUTPUT_DIR = ROOT / "outputs" / "error_compensation_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Figure settings
FIGURE_SIZE = (18, 5.5)
DPI = 300


# ============================================================
# Analysis function
# ============================================================

def analyze_system(system_name, file_path):

    print("\n" + "=" * 70)
    print(f"System: {system_name}")
    print("=" * 70)

    # Load predictions
    df = pd.read_parquet(file_path)

    # Prediction error: predicted value minus target value
    df["Pg_error"] = df["Pg_pred"] - df["Pg_target"]

    # Reshape data: one row per scenario, one column per bus
    errors = df.pivot(
        index="scenario",
        columns="bus",
        values="Pg_error"
    )

    # Extract errors at buses 0 and 1
    error_bus_0 = errors[0]
    error_bus_1 = errors[1]

    # Net error across the two buses
    net_error = error_bus_0 + error_bus_1

    # Sum of absolute errors
    absolute_error_sum = (
        error_bus_0.abs() + error_bus_1.abs()
    )

    # Error compensation percentage
    # Undefined when both errors are effectively zero
    compensation = (
        100
        * (
            1
            - net_error.abs()
            / absolute_error_sum
        )
    ).where(absolute_error_sum > 1e-12)

    # Correlation between bus-level errors
    pearson_correlation = error_bus_0.corr(error_bus_1)
    spearman_correlation = error_bus_0.corr(
        error_bus_1,
        method="spearman"
    )

    # Percentage of scenarios with opposite error signs
    opposite_signs = (
        error_bus_0 * error_bus_1 < 0
    ).mean() * 100

    # Net error statistics
    mean_absolute_net_error = net_error.abs().mean()
    p95_absolute_net_error = net_error.abs().quantile(0.95)
    maximum_absolute_net_error = net_error.abs().max()

    print(f"Number of scenarios: {len(errors):,}")
    print(f"Pearson correlation: {pearson_correlation:.5f}")
    print(f"Spearman correlation: {spearman_correlation:.5f}")
    print(f"Opposite-sign errors: {opposite_signs:.2f}%")
    print(
        "Mean compensation: "
        f"{compensation.mean():.2f}%"
    )
    print(
        "Median compensation: "
        f"{compensation.median():.2f}%"
    )
    print(
        "Mean absolute net error: "
        f"{mean_absolute_net_error:.6f} MW"
    )
    print(
        "95th percentile of absolute net error: "
        f"{p95_absolute_net_error:.6f} MW"
    )
    print(
        "Maximum absolute net error: "
        f"{maximum_absolute_net_error:.6f} MW"
    )

    # ========================================================
    # Create figures
    # ========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=FIGURE_SIZE
    )

    # --------------------------------------------------------
    # Figure 1: Bus-level error correlation
    # --------------------------------------------------------

    axes[0].scatter(
        error_bus_0,
        error_bus_1,
        s=9,
        alpha=0.25,
        edgecolors="none"
    )

    axes[0].axhline(
        0,
        color="black",
        linewidth=0.8
    )
    axes[0].axvline(
        0,
        color="black",
        linewidth=0.8
    )

    axes[0].set_xlabel(
        "Active Power Prediction Error at Bus 0 (MW)"
    )
    axes[0].set_ylabel(
        "Active Power Prediction Error at Bus 1 (MW)"
    )
    axes[0].set_title(
        f"{system_name}: Bus-Level Error Correlation\n"
        f"Pearson correlation = {pearson_correlation:.3f}"
    )
    axes[0].grid(True, alpha=0.2)

    # --------------------------------------------------------
    # Figure 2: Distribution of the net error
    # --------------------------------------------------------

    axes[1].hist(
        net_error,
        bins=70,
        edgecolor="white",
        linewidth=0.3
    )

    axes[1].axvline(
        0,
        color="black",
        linewidth=0.8
    )

    axes[1].set_xlabel(
        "Net Active Power Prediction Error (MW)"
    )
    axes[1].set_ylabel("Number of Scenarios")
    axes[1].set_title(
        f"{system_name}: Distribution of Net Error"
    )
    axes[1].grid(True, axis="y", alpha=0.2)

    # --------------------------------------------------------
    # Figure 3: Error compensation versus net error
    # --------------------------------------------------------

    valid = compensation.notna()

    axes[2].scatter(
        compensation[valid],
        net_error.abs()[valid],
        s=9,
        alpha=0.25,
        edgecolors="none"
    )

    axes[2].set_xlabel(
        "Error Compensation (%)"
    )
    axes[2].set_ylabel(
        "Absolute Net Active Power Prediction Error (MW)"
    )
    axes[2].set_title(
        f"{system_name}: Compensation vs. Net Error"
    )
    axes[2].set_xlim(-3, 103)
    axes[2].grid(True, alpha=0.2)

    # Overall figure formatting
    fig.suptitle(
        f"Active Power Prediction Error Analysis — {system_name}",
        fontsize=14,
        fontweight="bold"
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    # Save high-resolution figure
    filename = (
        OUTPUT_DIR
        / f"{system_name.lower().replace('-', '')}_error_analysis.png"
    )

    fig.savefig(
        filename,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Figure saved to: {filename}")

    # Export scenario-level results
    scenario_results = pd.DataFrame({
        "scenario": errors.index,
        "error_bus_0_MW": error_bus_0.to_numpy(),
        "error_bus_1_MW": error_bus_1.to_numpy(),
        "net_error_MW": net_error.to_numpy(),
        "absolute_net_error_MW": net_error.abs().to_numpy(),
        "compensation_pct": compensation.to_numpy(),
    })

    csv_path = (
        OUTPUT_DIR
        / f"{system_name.lower().replace('-', '')}_scenario_results.csv"
    )

    scenario_results.to_csv(
        csv_path,
        index=False
    )

    print(f"Scenario-level results saved to: {csv_path}")


# ============================================================
# Run analysis for both systems
# ============================================================

for system_name, file_path in SYSTEMS.items():

    if not file_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {file_path}"
        )

    analyze_system(
        system_name,
        file_path
    )


print("\n" + "=" * 70)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 70)
