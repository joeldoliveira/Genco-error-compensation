
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "outputs" / "extreme_scenario_analysis"
OUTPUT_DIR = INPUT_DIR / "error_relationships"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# VARIABLES
# ============================================================

VARIABLES = [
    "absolute_net_error_MW",
    "error_bus_0_MW",
    "error_bus_1_MW",
    "total_active_load_MW",
    "total_reactive_load_MVar",
]


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def analyze_system(system_name):

    print("\n" + "=" * 70)
    print(f"ERROR RELATIONSHIP ANALYSIS — {system_name.upper()}")
    print("=" * 70)

    input_file = INPUT_DIR / f"{system_name}_all_scenarios.csv"
    df = pd.read_csv(input_file)

    # Check that the required columns exist.
    required_columns = [
        "scenario",
        "absolute_net_error_MW",
        "total_active_load_MW",
        "total_reactive_load_MVar",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in {input_file.name}: {missing}"
        )

    # Calculate bus errors if they are not already available.
    # Prediction error = predicted generation - target generation.
    if "error_bus_0_MW" not in df.columns:
        df["error_bus_0_MW"] = (
            df["Pg_pred_bus_0"] - df["Pg_target_bus_0"]
        )

    if "error_bus_1_MW" not in df.columns:
        df["error_bus_1_MW"] = (
            df["Pg_pred_bus_1"] - df["Pg_target_bus_1"]
        )

    # --------------------------------------------------------
    # 1. CORRELATION MATRIX
    # --------------------------------------------------------

    available_variables = [
        col for col in VARIABLES if col in df.columns
    ]

    correlation = df[available_variables].corr(method="pearson")

    print("\nPearson correlation matrix:")
    print(correlation.round(4).to_string())

    correlation.to_csv(
        OUTPUT_DIR / f"{system_name}_correlation_matrix.csv"
    )

    # --------------------------------------------------------
    # 2. BUS ERROR RELATIONSHIP
    # --------------------------------------------------------

    error_0 = df["error_bus_0_MW"]
    error_1 = df["error_bus_1_MW"]

    correlation_buses = error_0.corr(error_1)
    opposite_sign_pct = (
        ((error_0 * error_1) < 0).mean() * 100
    )

    print("\nBus error relationship:")
    print(f"Pearson correlation: {correlation_buses:.4f}")
    print(f"Opposite signs: {opposite_sign_pct:.2f}%")

    # --------------------------------------------------------
    # 3. SCATTER PLOTS
    # --------------------------------------------------------

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Active load vs absolute net error.
    axes[0].scatter(
        df["total_active_load_MW"],
        df["absolute_net_error_MW"],
        alpha=0.35,
        s=12,
    )
    axes[0].set_xlabel("Total Active Load (MW)")
    axes[0].set_ylabel("Absolute Net Error (MW)")
    axes[0].set_title("Active Load vs Net Error")
    axes[0].grid(True, alpha=0.3)

    # Reactive load vs absolute net error.
    axes[1].scatter(
        df["total_reactive_load_MVar"],
        df["absolute_net_error_MW"],
        alpha=0.35,
        s=12,
    )
    axes[1].set_xlabel("Total Reactive Load (MVAr)")
    axes[1].set_ylabel("Absolute Net Error (MW)")
    axes[1].set_title("Reactive Load vs Net Error")
    axes[1].grid(True, alpha=0.3)

    # Bus 0 error vs bus 1 error.
    axes[2].scatter(
        error_0,
        error_1,
        alpha=0.35,
        s=12,
    )

    limits = [
        min(error_0.min(), error_1.min()),
        max(error_0.max(), error_1.max()),
    ]

    axes[2].plot(
        limits,
        [-limits[0], -limits[1]],
        linestyle="--",
        linewidth=1,
        label="Opposite-error reference",
    )

    axes[2].axhline(0, linewidth=0.8)
    axes[2].axvline(0, linewidth=0.8)
    axes[2].set_xlabel("Bus 0 Error (MW)")
    axes[2].set_ylabel("Bus 1 Error (MW)")
    axes[2].set_title("Bus 0 vs Bus 1 Errors")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"Error Relationships — {system_name.upper()}",
        fontsize=14,
    )
    fig.tight_layout()

    figure_file = OUTPUT_DIR / f"{system_name}_relationships.png"
    fig.savefig(figure_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------
    # 4. CORRELATION WITH LOADS
    # --------------------------------------------------------

    print("\nCorrelation with load variables:")

    for load_column in [
        "total_active_load_MW",
        "total_reactive_load_MVar",
    ]:
        r = df["absolute_net_error_MW"].corr(df[load_column])

        print(
            f"Absolute net error vs {load_column}: "
            f"{r:.4f}"
        )

    # --------------------------------------------------------
    # 5. SAVE SCENARIOS WITH THE LARGEST NET ERRORS
    # --------------------------------------------------------

    top_20 = df.nlargest(20, "absolute_net_error_MW")

    top_20.to_csv(
        OUTPUT_DIR / f"{system_name}_top_20_net_errors.csv",
        index=False,
    )

    print(f"\nScenarios analyzed: {len(df):,}")
    print(f"Correlation matrix saved.")
    print(f"Scatter plots saved: {figure_file}")
    print("Top 20 scenarios saved.")


# ============================================================
# RUN BOTH SYSTEMS
# ============================================================

if __name__ == "__main__":

    analyze_system("ieee14")
    analyze_system("ieee30")

    print("\nAnalysis completed.")
