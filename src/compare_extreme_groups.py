
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "outputs" / "extreme_scenario_analysis"
OUTPUT_DIR = INPUT_DIR / "group_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEMS = {
    "IEEE-14": INPUT_DIR / "ieee14_all_scenarios.csv",
    "IEEE-30": INPUT_DIR / "ieee30_all_scenarios.csv",
}

TOP_FRACTION = 0.01


def compare_groups(system_name, file_path):

    df = pd.read_csv(file_path)

    # Select exactly the highest-error 1% of scenarios
    n_extreme = max(1, int(np.ceil(TOP_FRACTION * len(df))))

    extreme_indices = df.nlargest(
        n_extreme,
        "absolute_net_error_MW",
    ).index

    df["scenario_group"] = "Remaining 99%"
    df.loc[extreme_indices, "scenario_group"] = "Worst 1%"

    metrics = [
        "absolute_net_error_MW",
        "absolute_error_bus_0_MW",
        "absolute_error_bus_1_MW",
        "total_active_load_MW",
        "total_reactive_load_MVar",
        "compensation_pct",
    ]

    rows = []

    for group_name, group in df.groupby("scenario_group"):

        for metric in metrics:

            values = group[metric].dropna()

            rows.append({
                "system": system_name,
                "group": group_name,
                "metric": metric,
                "n": len(group),
                "mean": values.mean(),
                "median": values.median(),
                "p95": values.quantile(0.95),
                "std": values.std(),
            })

    summary = pd.DataFrame(rows)

    summary.to_csv(
        OUTPUT_DIR / (
            f"{system_name.lower().replace('-', '')}_statistics.csv"
        ),
        index=False,
    )

    print(f"\n{'=' * 70}")
    print(f"EXTREME GROUP COMPARISON — {system_name}")
    print(f"{'=' * 70}")
    print(f"Total scenarios: {len(df):,}")
    print(f"Worst 1% scenarios: {n_extreme:,}")

    for metric in metrics:

        subset = summary[summary["metric"] == metric]

        print(f"\n{metric}")

        print(
            subset[["group", "mean", "median", "p95", "std"]]
            .to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    # Comparative boxplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    titles = [
        ("absolute_net_error_MW", "Absolute Net Error (MW)"),
        ("absolute_error_bus_0_MW", "Absolute Error at Bus 0 (MW)"),
        ("absolute_error_bus_1_MW", "Absolute Error at Bus 1 (MW)"),
        ("total_active_load_MW", "Total Active Load (MW)"),
        ("total_reactive_load_MVar", "Total Reactive Load (MVAr)"),
        ("compensation_pct", "Error Compensation (%)"),
    ]

    for ax, (column, title) in zip(axes.flat, titles):

        group_data = [
            df.loc[
                df["scenario_group"] == "Remaining 99%",
                column,
            ].dropna(),

            df.loc[
                df["scenario_group"] == "Worst 1%",
                column,
            ].dropna(),
        ]

        ax.boxplot(
            group_data,
            tick_labels=["Remaining 99%", "Worst 1%"],
            showfliers=False,
        )

        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"Extreme vs. Remaining Scenarios — {system_name}",
        fontsize=15,
        fontweight="bold",
    )

    fig.tight_layout()

    figure_path = OUTPUT_DIR / (
        f"{system_name.lower().replace('-', '')}_comparison.png"
    )

    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nStatistics saved to: {OUTPUT_DIR}")
    print(f"Figure saved to: {figure_path}")

    return summary


all_summaries = []

for system_name, file_path in SYSTEMS.items():

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required CSV not found: {file_path}"
        )

    all_summaries.append(
        compare_groups(system_name, file_path)
    )

pd.concat(all_summaries, ignore_index=True).to_csv(
    OUTPUT_DIR / "ieee14_ieee30_comparison.csv",
    index=False,
)

print("\nGroup comparison completed.")
