
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "outputs" / "extreme_scenario_analysis"
OUTPUT_DIR = INPUT_DIR / "extreme_visual_comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

SYSTEMS = ["ieee14", "ieee30"]

LOAD_COLUMNS = [
    "total_active_load_MW",
    "total_reactive_load_MVar",
]

ERROR_COLUMNS = [
    "error_bus_0_MW",
    "error_bus_1_MW",
]


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

def load_data(system):

    path = INPUT_DIR / f"{system}_all_scenarios.csv"
    df = pd.read_csv(path)

    # Calculate bus errors if not already present.
    # Prediction error = predicted generation - target generation.
    if "error_bus_0_MW" not in df.columns:
        df["error_bus_0_MW"] = (
            df["Pg_pred_bus_0"] - df["Pg_target_bus_0"]
        )

    if "error_bus_1_MW" not in df.columns:
        df["error_bus_1_MW"] = (
            df["Pg_pred_bus_1"] - df["Pg_target_bus_1"]
        )

    # Recalculate the net error from the two bus errors.
    df["net_error_MW"] = (
        df["error_bus_0_MW"] + df["error_bus_1_MW"]
    )

    df["absolute_net_error_MW"] = df["net_error_MW"].abs()

    # Identify exactly the 1% with the largest absolute net error.
    n_extreme = max(1, int(np.ceil(len(df) * 0.01)))

    extreme_indices = df.nlargest(
        n_extreme, "absolute_net_error_MW"
    ).index

    df["group"] = "Remaining 99%"
    df.loc[extreme_indices, "group"] = "Worst 1%"

    # Calculate error compensation.
    total_individual_error = (
        df["error_bus_0_MW"].abs()
        + df["error_bus_1_MW"].abs()
    )

    df["compensation_pct"] = np.where(
        total_individual_error > 0,
        100 * (
            1
            - df["absolute_net_error_MW"]
            / total_individual_error
        ),
        np.nan,
    )

    print(
        f"{system.upper()}: {len(df):,} scenarios; "
        f"{n_extreme:,} extreme scenarios"
    )

    return df


# ============================================================
# FIGURE 1: LOAD VS ERROR
# ============================================================

def plot_load_relationships(data, system):

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    groups = [
        ("Remaining 99%", "o"),
        ("Worst 1%", "^"),
    ]

    titles = [
        ("total_active_load_MW", "Total Active Load (MW)"),
        ("total_reactive_load_MVar", "Total Reactive Load (MVAr)"),
    ]

    for ax, (column, xlabel) in zip(axes, titles):

        for group, marker in groups:

            subset = data[data["group"] == group]

            ax.scatter(
                subset[column],
                subset["absolute_net_error_MW"],
                s=18 if group == "Worst 1%" else 10,
                alpha=0.75 if group == "Worst 1%" else 0.18,
                marker=marker,
                label=group,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Absolute Net Error (MW)")
        ax.set_title(f"{xlabel} vs Absolute Net Error")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(
        f"Extreme Scenario Comparison — {system.upper()}",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"{system}_load_vs_error.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# FIGURE 2: BUS ERROR COMPENSATION
# ============================================================

def plot_bus_errors(data, system):

    fig, ax = plt.subplots(figsize=(7, 6))

    for group, marker in [
        ("Remaining 99%", "o"),
        ("Worst 1%", "^"),
    ]:

        subset = data[data["group"] == group]

        ax.scatter(
            subset["error_bus_0_MW"],
            subset["error_bus_1_MW"],
            s=22 if group == "Worst 1%" else 10,
            alpha=0.8 if group == "Worst 1%" else 0.18,
            marker=marker,
            label=group,
        )

    # Reference line for complete cancellation:
    # error_bus_1 = -error_bus_0.
    values = np.concatenate([
        data["error_bus_0_MW"].to_numpy(),
        -data["error_bus_1_MW"].to_numpy(),
    ])

    bound = np.nanmax(np.abs(values))

    ax.plot(
        [-bound, bound],
        [bound, -bound],
        linestyle="--",
        linewidth=1.2,
        label="Complete cancellation reference",
    )

    ax.axhline(0, linewidth=0.7)
    ax.axvline(0, linewidth=0.7)

    ax.set_xlabel("Bus 0 Prediction Error (MW)")
    ax.set_ylabel("Bus 1 Prediction Error (MW)")
    ax.set_title(f"Bus Error Compensation — {system.upper()}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"{system}_bus_error_compensation.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# FIGURE 3: COMPENSATION VS NET ERROR
# ============================================================

def plot_compensation(data, system):

    fig, ax = plt.subplots(figsize=(8, 5))

    for group, marker in [
        ("Remaining 99%", "o"),
        ("Worst 1%", "^"),
    ]:

        subset = data[data["group"] == group]

        ax.scatter(
            subset["compensation_pct"],
            subset["absolute_net_error_MW"],
            s=22 if group == "Worst 1%" else 10,
            alpha=0.8 if group == "Worst 1%" else 0.18,
            marker=marker,
            label=group,
        )

    ax.set_xlabel("Error Compensation (%)")
    ax.set_ylabel("Absolute Net Error (MW)")
    ax.set_title(f"Compensation vs Net Error — {system.upper()}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"{system}_compensation_vs_error.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# SUMMARY TABLE
# ============================================================

def create_summary(data, system):

    metrics = [
        "absolute_net_error_MW",
        "error_bus_0_MW",
        "error_bus_1_MW",
        "total_active_load_MW",
        "total_reactive_load_MVar",
        "compensation_pct",
    ]

    rows = []

    for group, subset in data.groupby("group"):

        for metric in metrics:

            rows.append({
                "system": system.upper(),
                "group": group,
                "metric": metric,
                "mean": subset[metric].mean(),
                "median": subset[metric].median(),
                "p95": subset[metric].quantile(0.95),
            })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    summaries = []

    for system in SYSTEMS:

        df = load_data(system)

        plot_load_relationships(df, system)
        plot_bus_errors(df, system)
        plot_compensation(df, system)

        summary = create_summary(df, system)
        summaries.append(summary)

    final_summary = pd.concat(summaries, ignore_index=True)

    final_summary.to_csv(
        OUTPUT_DIR / "extreme_scenario_summary.csv",
        index=False,
    )

    print("\nAnalysis completed.")
    print(f"Results saved to: {OUTPUT_DIR}")
