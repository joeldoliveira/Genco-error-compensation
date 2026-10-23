
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

OUTPUT_DIR = ROOT / "outputs" / "extreme_scenario_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 20
EXTREME_PERCENTILE = 99


# ============================================================
# Extreme scenario analysis
# ============================================================

def analyze_extreme_scenarios(system_name, file_path):

    print("\n" + "=" * 75)
    print(f"Extreme Scenario Analysis — {system_name}")
    print("=" * 75)

    df = pd.read_parquet(file_path)

    # Prediction error: predicted value minus target value
    df["Pg_error"] = df["Pg_pred"] - df["Pg_target"]

    # Extract errors for buses 0 and 1
    errors = df.pivot(
        index="scenario",
        columns="bus",
        values="Pg_error"
    )

    error_bus_0 = errors[0]
    error_bus_1 = errors[1]

    # Net error and absolute errors
    net_error = error_bus_0 + error_bus_1
    absolute_net_error = net_error.abs()

    # Compensation percentage
    absolute_error_sum = error_bus_0.abs() + error_bus_1.abs()

    compensation = (
        100 * (1 - absolute_net_error / absolute_error_sum)
    ).where(absolute_error_sum > 1e-12)

    # Aggregate scenario-level load
    load_summary = df.groupby("scenario").agg(
        total_active_load_MW=("Pd", "sum"),
        total_reactive_load_MVar=("Qd", "sum"),
    )

    # Aggregate target and predicted generation, for context
    generation_summary = df.groupby("scenario").agg(
        total_target_generation_MW=("Pg_target", "sum"),
        total_predicted_generation_MW=("Pg_pred", "sum"),
    )

    # Assemble one row per scenario
    results = pd.DataFrame({
        "error_bus_0_MW": error_bus_0,
        "error_bus_1_MW": error_bus_1,
        "net_error_MW": net_error,
        "absolute_net_error_MW": absolute_net_error,
        "absolute_error_bus_0_MW": error_bus_0.abs(),
        "absolute_error_bus_1_MW": error_bus_1.abs(),
        "compensation_pct": compensation,
    })

    results = results.join(load_summary).join(generation_summary)
    results.index.name = "scenario"
    results = results.reset_index()

    # Rank scenarios by absolute net error
    results = results.sort_values(
        "absolute_net_error_MW",
        ascending=False
    )

    # Save the full scenario-level dataset
    full_path = OUTPUT_DIR / (
        f"{system_name.lower().replace('-', '')}_all_scenarios.csv"
    )
    results.to_csv(full_path, index=False)

    # Save the top N scenarios
    top_path = OUTPUT_DIR / (
        f"{system_name.lower().replace('-', '')}_top_{TOP_N}_scenarios.csv"
    )
    results.head(TOP_N).to_csv(top_path, index=False)

    # Identify the upper 1% of absolute net errors
    threshold = np.percentile(
        results["absolute_net_error_MW"],
        EXTREME_PERCENTILE
    )

    extreme = results[
        results["absolute_net_error_MW"] >= threshold
    ].copy()

    extreme_path = OUTPUT_DIR / (
        f"{system_name.lower().replace('-', '')}_extreme_1pct.csv"
    )
    extreme.to_csv(extreme_path, index=False)

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(f"Number of scenarios: {len(results):,}")
    print(f"99th percentile of absolute net error: {threshold:.6f} MW")
    print(
        "Maximum absolute net error: "
        f"{results['absolute_net_error_MW'].max():.6f} MW"
    )
    print(
        "Median absolute net error: "
        f"{results['absolute_net_error_MW'].median():.6f} MW"
    )
    print(
        "Correlation between absolute net error and total active load: "
        f"{results['absolute_net_error_MW'].corr(results['total_active_load_MW']):.4f}"
    )
    print(
        "Correlation between absolute net error and total reactive load: "
        f"{results['absolute_net_error_MW'].corr(results['total_reactive_load_MVar']):.4f}"
    )

    print(f"\nTop {TOP_N} scenarios by absolute net error:")
    print(
        results.head(TOP_N)[[
            "scenario",
            "error_bus_0_MW",
            "error_bus_1_MW",
            "net_error_MW",
            "absolute_net_error_MW",
            "compensation_pct",
            "total_active_load_MW",
            "total_reactive_load_MVar",
        ]].to_string(index=False)
    )

    # --------------------------------------------------------
    # Create diagnostic plots
    # --------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Absolute net error versus total active load
    axes[0].scatter(
        results["total_active_load_MW"],
        results["absolute_net_error_MW"],
        s=10,
        alpha=0.3,
        edgecolors="none",
    )
    axes[0].set_xlabel("Total Active Load (MW)")
    axes[0].set_ylabel("Absolute Net Prediction Error (MW)")
    axes[0].set_title(
        f"{system_name}: Net Error vs. Active Load"
    )
    axes[0].grid(True, alpha=0.2)

    # Absolute net error versus compensation
    valid = results["compensation_pct"].notna()

    axes[1].scatter(
        results.loc[valid, "compensation_pct"],
        results.loc[valid, "absolute_net_error_MW"],
        s=10,
        alpha=0.3,
        edgecolors="none",
    )
    axes[1].set_xlabel("Error Compensation (%)")
    axes[1].set_ylabel("Absolute Net Prediction Error (MW)")
    axes[1].set_title(
        f"{system_name}: Compensation vs. Net Error"
    )
    axes[1].grid(True, alpha=0.2)

    fig.suptitle(
        f"Extreme Scenario Analysis — {system_name}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    figure_path = OUTPUT_DIR / (
        f"{system_name.lower().replace('-', '')}_extreme_analysis.png"
    )
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFull results saved to: {full_path}")
    print(f"Top scenarios saved to: {top_path}")
    print(f"Extreme scenarios saved to: {extreme_path}")
    print(f"Diagnostic plots saved to: {figure_path}")


# ============================================================
# Run analysis
# ============================================================

for system_name, file_path in SYSTEMS.items():

    if not file_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {file_path}"
        )

    analyze_extreme_scenarios(system_name, file_path)


print("\n" + "=" * 75)
print("EXTREME SCENARIO ANALYSIS COMPLETED")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 75)



# ============================================================
# Comparison of extreme and remaining scenarios
# ============================================================

def compare_scenario_groups(system_name, results):

    results = results.copy()

    # Select exactly the scenarios in the top 1% by absolute net error
    n_extreme = max(1, int(np.ceil(0.01 * len(results))))

    extreme_indices = results.nlargest(
        n_extreme, "absolute_net_error_MW"
    ).index

    results["scenario_group"] = "Remaining 99%"
    results.loc[extreme_indices, "scenario_group"] = "Worst 1%"

    # Calculate group-level statistics
    rows = []

    metrics = {
        "Absolute net error (MW)": "absolute_net_error_MW",
        "Signed net error (MW)": "net_error_MW",
        "Total active load (MW)": "total_active_load_MW",
        "Total reactive load (MVAr)": "total_reactive_load_MVar",
        "Absolute error at Bus 0 (MW)": "absolute_error_bus_0_MW",
        "Absolute error at Bus 1 (MW)": "absolute_error_bus_1_MW",
        "Error compensation (%)": "compensation_pct",
    }

    for group_name, group in results.groupby("scenario_group"):

        for metric_name, column in metrics.items():

            values = group[column].dropna()

            rows.append({
                "system": system_name,
                "scenario_group": group_name,
                "metric": metric_name,
                "n_scenarios": len(group),
                "mean": values.mean(),
                "median": values.median(),
                "p95": values.quantile(0.95),
                "maximum": values.max(),
            })

        # Explicitly calculate RMSE of the signed net error
        rows.append({
            "system": system_name,
            "scenario_group": group_name,
            "metric": "Net error RMSE (MW)",
            "n_scenarios": len(group),
            "mean": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "maximum": np.sqrt(
                np.mean(group["net_error_MW"] ** 2)
            ),
        })

    summary = pd.DataFrame(rows)

    output_path = OUTPUT_DIR / (
        f"{system_name.lower().replace('-', '')}_"
        "extreme_vs_remaining_statistics.csv"
    )

    summary.to_csv(output_path, index=False)

    print(f"\nScenario group comparison — {system_name}")
    print(
        summary.pivot(
            index="metric",
            columns="scenario_group",
            values="mean",
        ).to_string(float_format=lambda x: f"{x:.5f}")
    )

    print(f"\nStatistics saved to: {output_path}")

    # --------------------------------------------------------
    # Comparative plots
    # --------------------------------------------------------

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # Plot 1: Active load distributions
    axes[0].boxplot(
        [
            results.loc[
                results["scenario_group"] == "Remaining 99%",
                "total_active_load_MW",
            ].dropna(),
            results.loc[
                results["scenario_group"] == "Worst 1%",
                "total_active_load_MW",
            ].dropna(),
        ],
        tick_labels=["Remaining 99%", "Worst 1%"],
        showfliers=False,
    )
    axes[0].set_title("Active Load by Scenario Group")
    axes[0].set_ylabel("Total Active Load (MW)")
    axes[0].grid(True, axis="y", alpha=0.25)

    # Plot 2: Absolute net error distributions
    axes[1].boxplot(
        [
            results.loc[
                results["scenario_group"] == "Remaining 99%",
                "absolute_net_error_MW",
            ].dropna(),
            results.loc[
                results["scenario_group"] == "Worst 1%",
                "absolute_net_error_MW",
            ].dropna(),
        ],
        tick_labels=["Remaining 99%", "Worst 1%"],
        showfliers=False,
    )
    axes[1].set_title("Absolute Net Error by Scenario Group")
    axes[1].set_ylabel("Absolute Net Prediction Error (MW)")
    axes[1].grid(True, axis="y", alpha=0.25)

    # Plot 3: Compensation distributions
    axes[2].boxplot(
        [
            results.loc[
                results["scenario_group"] == "Remaining 99%",
                "compensation_pct",
            ].dropna(),
            results.loc[
                results["scenario_group"] == "Worst 1%",
                "compensation_pct",
            ].dropna(),
        ],
        tick_labels=["Remaining 99%", "Worst 1%"],
        showfliers=False,
    )
    axes[2].set_title("Error Compensation by Scenario Group")
    axes[2].set_ylabel("Error Compensation (%)")
    axes[2].grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        f"Extreme vs. Remaining Scenarios — {system_name}",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout()

    figure_path = OUTPUT_DIR / (
        f"{system_name.lower().replace('-', '')}_"
        "extreme_vs_remaining.png"
    )

    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Comparison plot saved to: {figure_path}")
