import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURAÇÃO
# ============================================================

FILE = Path(
    "outputs/case30_seed42/case30_ieee_predictions.parquet"
)

OUTPUT = Path("outputs/case30_seed42")


# ============================================================
# CARREGAR RESULTADOS
# ============================================================

print("A carregar resultados do GENCO...")

df = pd.read_parquet(FILE)

print(f"\nDataset carregado.")
print(f"Shape: {df.shape}")


# ============================================================
# ERROS ABSOLUTOS
# ============================================================

df["error_Vm"] = np.abs(
    df["Vm_pred"] - df["Vm_target"]
)

df["error_Va"] = np.abs(
    df["Va_pred"] - df["Va_target"]
)

df["error_Pg"] = np.abs(
    df["Pg_pred"] - df["Pg_target"]
)

df["error_Qg"] = np.abs(
    df["Qg_pred"] - df["Qg_target"]
)


# ============================================================
# ERRO POR BUS
# ============================================================

bus_errors = (
    df.groupby("bus")
    .agg(
        N=("scenario", "size"),

        MAE_Vm=("error_Vm", "mean"),
        RMSE_Vm=("error_Vm",
                 lambda x: np.sqrt(np.mean(x**2))),
        P99_Vm=("error_Vm",
                lambda x: x.quantile(0.99)),
        MAX_Vm=("error_Vm", "max"),

        MAE_Va=("error_Va", "mean"),
        RMSE_Va=("error_Va",
                 lambda x: np.sqrt(np.mean(x**2))),
        P99_Va=("error_Va",
                lambda x: x.quantile(0.99)),
        MAX_Va=("error_Va", "max"),

        MAE_Pg=("error_Pg", "mean"),
        P99_Pg=("error_Pg",
                lambda x: x.quantile(0.99)),
        MAX_Pg=("error_Pg", "max"),

        MAE_Qg=("error_Qg", "mean"),
        P99_Qg=("error_Qg",
                lambda x: x.quantile(0.99)),
        MAX_Qg=("error_Qg", "max"),
    )
)


# ============================================================
# RESULTADOS
# ============================================================

print("\n" + "=" * 70)
print("ERRO POR BUS - IEEE 30")
print("=" * 70)

print("\n")
print(bus_errors.round(6))


# ============================================================
# TOP 5 BUSES - Vm
# ============================================================

print("\n" + "=" * 70)
print("TOP 5 BUSES - MAIOR MAE DE Vm")
print("=" * 70)

print(
    bus_errors
    .sort_values("MAE_Vm", ascending=False)
    [["MAE_Vm", "P99_Vm", "MAX_Vm"]]
    .head(5)
    .round(6)
)


# ============================================================
# TOP 5 BUSES - Va
# ============================================================

print("\n" + "=" * 70)
print("TOP 5 BUSES - MAIOR MAE DE Va")
print("=" * 70)

print(
    bus_errors
    .sort_values("MAE_Va", ascending=False)
    [["MAE_Va", "P99_Va", "MAX_Va"]]
    .head(5)
    .round(6)
)


# ============================================================
# TOP 5 BUSES - Pg
# ============================================================

print("\n" + "=" * 70)
print("TOP 5 BUSES - MAIOR MAE DE Pg")
print("=" * 70)

print(
    bus_errors
    .sort_values("MAE_Pg", ascending=False)
    [["MAE_Pg", "P99_Pg", "MAX_Pg"]]
    .head(5)
    .round(6)
)


# ============================================================
# TOP 5 BUSES - Qg
# ============================================================

print("\n" + "=" * 70)
print("TOP 5 BUSES - MAIOR MAE DE Qg")
print("=" * 70)

print(
    bus_errors
    .sort_values("MAE_Qg", ascending=False)
    [["MAE_Qg", "P99_Qg", "MAX_Qg"]]
    .head(5)
    .round(6)
)


# ============================================================
# CORRELAÇÃO - PONTO 1
# ANÁLISE AO NÍVEL DO CENÁRIO
# ============================================================

print("\n" + "=" * 70)
print("CORRELAÇÃO ENTRE STRESS DO CENÁRIO E ERRO")
print("=" * 70)


# ============================================================
# 1. AGREGAR OS DADOS POR CENÁRIO
# ============================================================

scenario_analysis = (
    df.groupby("scenario")
    .agg(
        # Stress do cenário
        Pd_total=("Pd", "sum"),
        Qd_total=("Qd", "sum"),
        Vm_min=("Vm_target", "min"),

        # Erros médios do GENCO
        MAE_Vm=("error_Vm", "mean"),
        MAE_Va=("error_Va", "mean"),
        MAE_Pg=("error_Pg", "mean"),
        MAE_Qg=("error_Qg", "mean"),
    )
)


print("\nNúmero de cenários analisados:")
print(len(scenario_analysis))


# ============================================================
# 2. VARIÁVEIS DE STRESS E ERROS
# ============================================================

stress_variables = [
    "Pd_total",
    "Qd_total",
    "Vm_min"
]

error_variables = [
    "MAE_Vm",
    "MAE_Va",
    "MAE_Pg",
    "MAE_Qg"
]


# ============================================================
# 3. CALCULAR CORRELAÇÃO
# ============================================================

scenario_correlation = (
    scenario_analysis[
        stress_variables + error_variables
    ].corr()
)


# ============================================================
# 4. MOSTRAR MATRIZ
# ============================================================

print("\nMatriz de correlação:")

print(
    scenario_correlation
    .loc[stress_variables, error_variables]
    .round(4)
)


# ============================================================
# 5. MOSTRAR CORRELAÇÕES INDIVIDUAIS
# ============================================================

print("\n" + "=" * 70)
print("CORRELAÇÃO ENTRE STRESS E ERRO")
print("=" * 70)

for error in error_variables:

    print(f"\n{error}:")

    for variable in stress_variables:

        value = scenario_correlation.loc[
            variable,
            error
        ]

        print(
            f"  {variable:10s} -> "
            f"{value:+.4f}"
        )


# ============================================================
# 6. GUARDAR RESULTADOS
# ============================================================

correlation_file = (
    OUTPUT / "case30_scenario_correlation.csv"
)

scenario_correlation.to_csv(
    correlation_file
)


print("\n" + "=" * 70)
print("ANÁLISE DE CORRELAÇÃO TERMINADA")
print("=" * 70)

print("\nFicheiro criado:")
print(correlation_file)

# ============================================================
# PONTO 2 - NÍVEIS DE STRESS
# ============================================================

print("\n" + "=" * 70)
print("NÍVEIS DE STRESS DOS CENÁRIOS")
print("=" * 70)

# Dividir os cenários em 3 grupos segundo a tensão mínima
scenario_analysis["stress_voltage"] = pd.cut(
    scenario_analysis["Vm_min"],
    bins=[-np.inf, 0.96, 0.98, np.inf],
    labels=[
        "Stress alto",
        "Stress médio",
        "Stress baixo"
    ]
)

print("\nNúmero de cenários por nível de stress:")

print(
    scenario_analysis["stress_voltage"]
    .value_counts()
    .sort_index()
)

# ============================================================
# PONTO 3 - ERRO MÉDIO POR NÍVEL DE STRESS
# ============================================================

print("\n" + "=" * 70)
print("ERRO MÉDIO POR NÍVEL DE STRESS")
print("=" * 70)

stress_errors = (
    scenario_analysis
    .groupby("stress_voltage", observed=True)
    .agg(
        N=("MAE_Vm", "size"),
        MAE_Vm=("MAE_Vm", "mean"),
        MAE_Va=("MAE_Va", "mean"),
        MAE_Pg=("MAE_Pg", "mean"),
        MAE_Qg=("MAE_Qg", "mean"),
    )
)

print("\n")
print(stress_errors.round(6))

# ============================================================
# PONTO 4 - COMPARAÇÃO RELATIVA DO ERRO
# ============================================================

print("\n" + "=" * 70)
print("AUMENTO DO ERRO: STRESS ALTO VS STRESS BAIXO")
print("=" * 70)

high = stress_errors.loc["Stress alto"]
low = stress_errors.loc["Stress baixo"]

for error in ["MAE_Vm", "MAE_Va", "MAE_Pg", "MAE_Qg"]:

    ratio = high[error] / low[error]

    print(
        f"{error:8s} -> "
        f"{ratio:.2f}x"
    )

    # ============================================================
# PONTO 5 - DISTRIBUIÇÃO DO ERRO POR NÍVEL DE STRESS
# ============================================================

print("\n" + "=" * 70)
print("PERCENTIS DO ERRO POR NÍVEL DE STRESS")
print("=" * 70)

for stress in ["Stress alto", "Stress médio", "Stress baixo"]:

    data = scenario_analysis[
        scenario_analysis["stress_voltage"] == stress
    ]

    print(f"\n{stress}:")

    for error in ["MAE_Vm", "MAE_Va", "MAE_Pg", "MAE_Qg"]:

        p50 = data[error].quantile(0.50)
        p95 = data[error].quantile(0.95)
        p99 = data[error].quantile(0.99)
        max_error = data[error].max()

        print(
            f"{error:8s} | "
            f"P50={p50:.6f} | "
            f"P95={p95:.6f} | "
            f"P99={p99:.6f} | "
            f"MAX={max_error:.6f}"
        )

        # ============================================================
# PONTO 6 - PIORES CENÁRIOS DE STRESS ALTO
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 CENÁRIOS - STRESS ALTO E MAIOR ERRO DE Pg")
print("=" * 70)

top_stress_pg = (
    scenario_analysis[
        scenario_analysis["stress_voltage"] == "Stress alto"
    ]
    .sort_values("MAE_Pg", ascending=False)
    [["Pd_total", "Qd_total", "Vm_min",
      "MAE_Vm", "MAE_Va", "MAE_Pg", "MAE_Qg"]]
    .head(10)
)

print(top_stress_pg.round(6))

# ============================================================
# PONTO 7 - CORRELAÇÃO DENTRO DO STRESS ALTO
# ============================================================

print("\n" + "=" * 70)
print("CORRELAÇÃO: STRESS ALTO")
print("=" * 70)

high_stress = scenario_analysis[
    scenario_analysis["stress_voltage"] == "Stress alto"
]

corr_high = high_stress[
    ["Pd_total", "Qd_total", "Vm_min",
     "MAE_Vm", "MAE_Va", "MAE_Pg", "MAE_Qg"]
].corr()

print(corr_high.round(4))

# ============================================================
# PONTO 8 - ERRO DE Pg EM FUNÇÃO DE Vm_min
# ============================================================

print("\n" + "=" * 70)
print("ERRO DE Pg EM FUNÇÃO DE Vm_min - STRESS ALTO")
print("=" * 70)

high_stress = scenario_analysis[
    scenario_analysis["stress_voltage"] == "Stress alto"
].copy()

# Criar grupos de Vm_min
high_stress["Vm_group"] = pd.cut(
    high_stress["Vm_min"],
    bins=[-np.inf, 0.94, 0.95, 0.96],
    labels=[
        "Vm_min <= 0.94",
        "0.94 < Vm_min <= 0.95",
        "0.95 < Vm_min < 0.96"
    ]
)

result_vm = (
    high_stress
    .groupby("Vm_group", observed=True)
    .agg(
        N=("MAE_Pg", "size"),
        Vm_min_medio=("Vm_min", "mean"),
        MAE_Pg=("MAE_Pg", "mean"),
        P95_Pg=("MAE_Pg", lambda x: x.quantile(0.95)),
        P99_Pg=("MAE_Pg", lambda x: x.quantile(0.99)),
        MAX_Pg=("MAE_Pg", "max")
    )
)

print(result_vm.round(6))

# ============================================================
# PONTO 9 - QUAIS OS BUSES COM MAIOR ERRO DE Pg
#            NOS CENÁRIOS MAIS SEVEROS
# ============================================================

print("\n" + "=" * 70)
print("ERRO DE Pg POR BUS - Vm_min <= 0.94")
print("=" * 70)

severe_scenarios = scenario_analysis[
    scenario_analysis["Vm_min"] <= 0.94
].index

df_severe = df[df["scenario"].isin(severe_scenarios)].copy()

bus_pg_severe = (
    df_severe
    .groupby("bus")
    .agg(
        N=("error_Pg", "size"),
        MAE_Pg=("error_Pg", "mean"),
        P95_Pg=("error_Pg", lambda x: x.quantile(0.95)),
        P99_Pg=("error_Pg", lambda x: x.quantile(0.99)),
        MAX_Pg=("error_Pg", "max")
    )
    .sort_values("MAE_Pg", ascending=False)
)

print(bus_pg_severe.head(10).round(6))

# ============================================================
# PONTO 10 - Pg TARGET VS PRED NOS BUSES 0 E 1
#             NOS CENÁRIOS MAIS SEVEROS
# ============================================================

print("\n" + "=" * 70)
print("Pg TARGET VS PRED - BUSES 0 E 1 - Vm_min <= 0.94")
print("=" * 70)

df_pg_severe = df_severe[
    df_severe["bus"].isin([0, 1])
].copy()

pg_comparison = (
    df_pg_severe
    .groupby("bus")
    .agg(
        N=("Pg_target", "size"),
        Pg_target_mean=("Pg_target", "mean"),
        Pg_pred_mean=("Pg_pred", "mean"),
        Pg_target_min=("Pg_target", "min"),
        Pg_target_max=("Pg_target", "max"),
        Pg_pred_min=("Pg_pred", "min"),
        Pg_pred_max=("Pg_pred", "max"),
        MAE_Pg=("error_Pg", "mean"),
        P95_Pg=("error_Pg", lambda x: x.quantile(0.95)),
        MAX_Pg=("error_Pg", "max")
    )
)

print(pg_comparison.round(6))

# ============================================================
# PONTO 11 - CASOS COM MAIOR ERRO DE Pg
#             NOS BUSES 0 E 1
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 MAIORES ERROS DE Pg - BUSES 0 E 1")
print("=" * 70)

top_pg_errors = (
    df_pg_severe[
        ["scenario", "bus", "Pg_target", "Pg_pred", "error_Pg", "Vm_target"]
    ]
    .sort_values("error_Pg", ascending=False)
    .head(10)
)

print(top_pg_errors.to_string(index=False))

# ============================================================
# PONTO 12 - ERRO DE GERAÇÃO TOTAL
#             BUSES 0 E 1 - CENÁRIOS SEVEROS
# ============================================================

print("\n" + "=" * 70)
print("ERRO DE GERAÇÃO TOTAL - BUSES 0 E 1")
print("=" * 70)

pg_total = (
    df_pg_severe
    .groupby("scenario")
    .agg(
        Pg_target_total=("Pg_target", "sum"),
        Pg_pred_total=("Pg_pred", "sum")
    )
)

pg_total["error_total"] = np.abs(
    pg_total["Pg_pred_total"] - pg_total["Pg_target_total"]
)

print(pg_total["error_total"].describe().round(6))

print("\nTOP 10 MAIORES ERROS DE GERAÇÃO TOTAL:")

print(
    pg_total
    .sort_values("error_total", ascending=False)
    .head(10)
    .round(6)
)

# ============================================================
# PONTO 13 - COMPENSAÇÃO DO ERRO ENTRE OS BUSES 0 E 1
# ============================================================

print("\n" + "=" * 70)
print("COMPENSAÇÃO DOS ERROS DE Pg - BUSES 0 E 1")
print("=" * 70)

pg_errors = (
    df_pg_severe[
        ["scenario", "bus", "error_Pg"]
    ]
    .pivot(
        index="scenario",
        columns="bus",
        values="error_Pg"
    )
)

# Erro com sinal, e não em valor absoluto
df_signed = df_pg_severe.copy()
df_signed["signed_error_Pg"] = (
    df_signed["Pg_pred"] - df_signed["Pg_target"]
)

signed_errors = (
    df_signed[
        ["scenario", "bus", "signed_error_Pg"]
    ]
    .pivot(
        index="scenario",
        columns="bus",
        values="signed_error_Pg"
    )
)

signed_errors.columns = ["error_bus_0", "error_bus_1"]

print("\nCorrelação entre os erros dos dois geradores:")
print(
    signed_errors["error_bus_0"]
    .corr(signed_errors["error_bus_1"])
)

print("\nMédia dos erros com sinal:")

print(
    signed_errors.mean().round(6)
)

print("\nTOP 10 cenários com maior compensação:")

signed_errors["error_sum"] = (
    signed_errors["error_bus_0"] +
    signed_errors["error_bus_1"]
)

print(
    signed_errors
    .sort_values("error_sum", key=lambda x: np.abs(x), ascending=False)
    .head(10)
    .round(6)
)

# ============================================================
# PONTO 14 - QUANTIFICAR A COMPENSAÇÃO
# ============================================================

print("\n" + "=" * 70)
print("QUANTIFICAÇÃO DA COMPENSAÇÃO DOS ERROS")
print("=" * 70)

# Soma dos módulos dos erros individuais
individual_error = (
    np.abs(signed_errors["error_bus_0"]) +
    np.abs(signed_errors["error_bus_1"])
)

# Módulo do erro depois da compensação
total_error = np.abs(
    signed_errors["error_sum"]
)

# Percentagem de compensação
compensation = (
    1 - total_error / individual_error
) * 100

print("\nPercentagem de compensação:")
print(compensation.describe().round(2))

print("\nCompensação média:")
print(f"{compensation.mean():.2f}%")

print("\nCompensação mediana:")
print(f"{compensation.median():.2f}%")

# ============================================================
# PONTO 15 - ERRO INDIVIDUAL VS ERRO TOTAL
# ============================================================

print("\n" + "=" * 70)
print("ERRO INDIVIDUAL VS ERRO TOTAL - Pg")
print("=" * 70)

print("\nErro absoluto individual:")

print(
    individual_error.describe().round(6)
)

print("\nErro absoluto após compensação:")

print(
    total_error.describe().round(6)
)

print("\nMédias:")

print(
    f"Erro individual médio (soma dos módulos): "
    f"{individual_error.mean():.6f} MW"
)

print(
    f"Erro total médio (após compensação): "
    f"{total_error.mean():.6f} MW"
)

print(
    f"Redução média do erro: "
    f"{(1 - total_error.mean() / individual_error.mean()) * 100:.2f}%"
)

# ============================================================
# PONTO 16 - COMPENSAÇÃO VS NÍVEL DE STRESS
# ============================================================

print("\n" + "=" * 70)
print("COMPENSAÇÃO DOS ERROS VS NÍVEL DE STRESS")
print("=" * 70)

compensation_df = pd.DataFrame({
    "stress": scenario_analysis.loc[signed_errors.index, "stress_voltage"],
    "individual_error": individual_error,
    "total_error": total_error,
    "compensation": compensation
})

stress_compensation = (
    compensation_df
    .groupby("stress", observed=True)
    .agg(
        N=("compensation", "size"),
        erro_individual_medio=("individual_error", "mean"),
        erro_total_medio=("total_error", "mean"),
        compensacao_media=("compensation", "mean"),
        compensacao_mediana=("compensation", "median")
    )
)

print(stress_compensation.round(3))

# ============================================================
# PONTO 17 - COMPENSAÇÃO EM TODOS OS CENÁRIOS
# ============================================================

print("\n" + "=" * 70)
print("COMPENSAÇÃO DOS ERROS DE Pg - TODOS OS CENÁRIOS")
print("=" * 70)

df_pg_all = df[df["bus"].isin([0, 1])].copy()

signed_all = (
    df_pg_all
    .assign(
        signed_error_Pg=lambda x: x["Pg_pred"] - x["Pg_target"]
    )
    [["scenario", "bus", "signed_error_Pg"]]
    .pivot(
        index="scenario",
        columns="bus",
        values="signed_error_Pg"
    )
)

signed_all.columns = ["error_bus_0", "error_bus_1"]

individual_all = (
    np.abs(signed_all["error_bus_0"]) +
    np.abs(signed_all["error_bus_1"])
)

total_all = np.abs(
    signed_all["error_bus_0"] +
    signed_all["error_bus_1"]
)

compensation_all = (
    1 - total_all / individual_all
) * 100

compensation_all = compensation_all.replace(
    [np.inf, -np.inf], np.nan
).dropna()

print("\nNúmero de cenários:")
print(len(compensation_all))

print("\nCompensação:")
print(
    compensation_all.describe().round(2)
)

print("\nCompensação média:")
print(f"{compensation_all.mean():.2f}%")

print("\nCompensação mediana:")
print(f"{compensation_all.median():.2f}%")

# ============================================================
# PONTO 18 - ERRO TOTAL DE Pg VS STRESS
# ============================================================

print("\n" + "=" * 70)
print("ERRO TOTAL DE Pg VS NÍVEL DE STRESS")
print("=" * 70)

pg_all = (
    df_pg_all
    .groupby("scenario")
    .agg(
        Pg_target_total=("Pg_target", "sum"),
        Pg_pred_total=("Pg_pred", "sum")
    )
)

pg_all["error_total"] = np.abs(
    pg_all["Pg_pred_total"] - pg_all["Pg_target_total"]
)

pg_all["stress"] = scenario_analysis.loc[
    pg_all.index, "stress_voltage"
]

total_error_stress = (
    pg_all
    .groupby("stress", observed=True)
    .agg(
        N=("error_total", "size"),
        MAE_total_Pg=("error_total", "mean"),
        P50_total_Pg=("error_total", "median"),
        P95_total_Pg=("error_total", lambda x: x.quantile(0.95)),
        P99_total_Pg=("error_total", lambda x: x.quantile(0.99)),
        MAX_total_Pg=("error_total", "max")
    )
)

print(
    total_error_stress.round(6)
)

# ============================================================
# PONTO 19 - O QUE EXPLICA O ERRO TOTAL DE Pg?
# ============================================================

print("\n" + "=" * 70)
print("CORRELAÇÃO DO ERRO TOTAL DE Pg")
print("=" * 70)

correlation_total = pg_all.join(
    scenario_analysis[
        ["Pd_total", "Qd_total", "Vm_min"]
    ]
)

print(
    correlation_total[
        [
            "error_total",
            "Pd_total",
            "Qd_total",
            "Vm_min"
        ]
    ]
    .corr()["error_total"]
    .round(4)
)

# ============================================================
# PONTO 20 - ERRO TOTAL DE Pg EM FUNÇÃO DE Vm_min
# ============================================================

print("\n" + "=" * 70)
print("ERRO TOTAL DE Pg EM FUNÇÃO DE Vm_min")
print("=" * 70)

pg_all["Vm_min"] = scenario_analysis.loc[
    pg_all.index,
    "Vm_min"
]

pg_all["Vm_group"] = pd.cut(
    pg_all["Vm_min"],
    bins=[-np.inf, 0.94, 0.95, 0.96, np.inf],
    labels=[
        "Vm_min <= 0.94",
        "0.94 < Vm_min <= 0.95",
        "0.95 < Vm_min < 0.96",
        "Vm_min >= 0.96"
    ]
)

vm_total_error = (
    pg_all
    .groupby("Vm_group", observed=True)
    .agg(
        N=("error_total", "size"),
        MAE_total_Pg=("error_total", "mean"),
        P50_total_Pg=("error_total", "median"),
        P95_total_Pg=("error_total", lambda x: x.quantile(0.95)),
        P99_total_Pg=("error_total", lambda x: x.quantile(0.99)),
        MAX_total_Pg=("error_total", "max")
    )
)

print(
    vm_total_error.round(6)
)


# ============================================================
# PONTO 21 - GRÁFICO DO ERRO TOTAL DE Pg
# ============================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(9, 5))

plt.bar(
    vm_total_error.index.astype(str),
    vm_total_error["MAE_total_Pg"]
)

plt.xlabel("Intervalo de Vm_min (pu)")
plt.ylabel("MAE do erro total de Pg (MW)")
plt.title("Erro total de Pg em função da tensão mínima")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


#Vamos então começar a analisar Qg.


# ============================================================
# PONTO 22 - ANÁLISE DO ERRO DE Qg POR BUS
# ============================================================

print("\n" + "=" * 70)
print("ERRO DE Qg POR BUS")
print("=" * 70)

bus_qg_analysis = (
    df
    .groupby("bus")
    .agg(
        N=("error_Qg", "size"),
        MAE_Qg=("error_Qg", "mean"),
        RMSE_Qg=("error_Qg", lambda x: np.sqrt(np.mean(x**2))),
        P95_Qg=("error_Qg", lambda x: x.quantile(0.95)),
        P99_Qg=("error_Qg", lambda x: x.quantile(0.99)),
        MAX_Qg=("error_Qg", "max"),
        Qg_target_mean=("Qg_target", "mean"),
        Qg_pred_mean=("Qg_pred", "mean")
    )
    .sort_values("MAE_Qg", ascending=False)
)

print(
    bus_qg_analysis.round(6)
)

# ============================================================
# PONTO 23 - ERRO COM SINAL DE Qg POR BUS
# ============================================================

print("\n" + "=" * 70)
print("ERRO COM SINAL DE Qg POR BUS")
print("=" * 70)

df["signed_error_Qg"] = (
    df["Qg_pred"] - df["Qg_target"]
)

signed_qg_bus = (
    df
    .groupby("bus")
    .agg(
        mean_signed_error_Qg=("signed_error_Qg", "mean"),
        median_signed_error_Qg=("signed_error_Qg", "median"),
        Qg_target_mean=("Qg_target", "mean"),
        Qg_pred_mean=("Qg_pred", "mean")
    )
    .sort_values(
        "mean_signed_error_Qg",
        ascending=False
    )
)

print(
    signed_qg_bus.round(6)
)

# ============================================================
# PONTO 24 - CORRELAÇÃO DOS ERROS DE Qg ENTRE GERADORES
# ============================================================

print("\n" + "=" * 70)
print("CORRELAÇÃO DOS ERROS DE Qg ENTRE GERADORES")
print("=" * 70)

qg_generators = df[df["Qg_target"] != 0].copy()

signed_qg = (
    qg_generators
    .pivot(
        index="scenario",
        columns="bus",
        values="signed_error_Qg"
    )
)

print("\nCorrelação entre os erros com sinal:")

print(
    signed_qg.corr().round(4)
)

# ============================================================
# PONTO 25 - COMPENSAÇÃO DOS ERROS DE Qg
# ============================================================

print("\n" + "=" * 70)
print("COMPENSAÇÃO DOS ERROS DE Qg")
print("=" * 70)

# Erro absoluto individual de cada gerador
qg_abs = signed_qg.abs()

# Soma dos erros absolutos individuais
individual_qg_error = qg_abs.sum(axis=1)

# Erro resultante depois da compensação entre geradores
total_qg_error = signed_qg.sum(axis=1).abs()

# Percentagem de compensação
compensation_qg = (
    1 - total_qg_error / individual_qg_error
) * 100

# Evitar problemas quando o erro individual é zero
compensation_qg = compensation_qg.replace([np.inf, -np.inf], np.nan)

print("\nCompensação dos erros de Qg:")

print(
    compensation_qg.describe()
)

print("\nMédia de erro Qg individual:")
print(individual_qg_error.mean())

print("\nMédia de erro Qg total após compensação:")
print(total_qg_error.mean())

print("\nMédia de compensação:")
print(compensation_qg.mean())

print("\nMediana de compensação:")
print(compensation_qg.median())

# ============================================================
# PONTO 26 - COMPENSAÇÃO DE Qg VS STRESS DE TENSÃO
# ============================================================

qg_compensation_df = pd.DataFrame({
    "scenario": compensation_qg.index,
    "compensation_Qg": compensation_qg.values
})

qg_compensation_df = qg_compensation_df.merge(
    scenario_analysis[["Vm_min", "stress_voltage"]],
    left_on="scenario",
    right_index=True
)

print("\n" + "=" * 70)
print("COMPENSAÇÃO DE Qg VS STRESS DE TENSÃO")
print("=" * 70)

print(
    qg_compensation_df
    .groupby("stress_voltage")["compensation_Qg"]
    .agg(
        N="count",
        mean="mean",
        median="median",
        P25=lambda x: x.quantile(0.25),
        P75=lambda x: x.quantile(0.75),
        min="min",
        max="max"
    )
)

# ============================================================
# PONTO 27 - ERRO TOTAL DE Qg VS STRESS
# ============================================================

qg_total_stress = pd.DataFrame({
    "scenario": total_qg_error.index,
    "total_Qg_error": total_qg_error.values
})

qg_total_stress = qg_total_stress.merge(
    scenario_analysis[["Vm_min", "stress_voltage"]],
    left_on="scenario",
    right_index=True
)

print("\n" + "=" * 70)
print("ERRO TOTAL DE Qg VS STRESS")
print("=" * 70)

print(
    qg_total_stress
    .groupby("stress_voltage")["total_Qg_error"]
    .agg(
        N="count",
        MAE="mean",
        P50="median",
        P95=lambda x: x.quantile(0.95),
        P99=lambda x: x.quantile(0.99),
        MAX="max"
    )
)

# ============================================================
# PONTO 28 - ERRO TOTAL DE Qg VS Vm_min
# ============================================================

qg_total_stress["Vm_group"] = pd.cut(
    qg_total_stress["Vm_min"],
    bins=[-np.inf, 0.94, 0.95, 0.96, np.inf],
    labels=[
        "Vm_min <= 0.94",
        "0.94 < Vm_min <= 0.95",
        "0.95 < Vm_min < 0.96",
        "Vm_min >= 0.96"
    ]
)

print("\n" + "=" * 70)
print("ERRO TOTAL DE Qg VS Vm_min")
print("=" * 70)

print(
    qg_total_stress
    .groupby("Vm_group", observed=False)["total_Qg_error"]
    .agg(
        N="count",
        MAE="mean",
        P50="median",
        P95=lambda x: x.quantile(0.95),
        P99=lambda x: x.quantile(0.99),
        MAX="max"
    )
)

# ============================================================
# PONTO 29 - CORRELAÇÃO Vm_min VS ERRO TOTAL DE Qg
# ============================================================

print("\n" + "=" * 70)
print("CORRELAÇÃO Vm_min VS ERRO TOTAL DE Qg")
print("=" * 70)

print(
    qg_total_stress[
        ["Vm_min", "total_Qg_error"]
    ].corr()
)

# ============================================================
# PONTO 30 - TOP CENÁRIOS PELO ERRO TOTAL DE Qg
# ============================================================

top_qg_scenarios = (
    qg_total_stress
    .sort_values("total_Qg_error", ascending=False)
    .head(10)
)

print("\n" + "=" * 70)
print("TOP 10 CENÁRIOS PELO ERRO TOTAL DE Qg")
print("=" * 70)

print(
    top_qg_scenarios[
        [
            "total_Qg_error",
            "Vm_min",
            "stress_voltage"
        ]
    ]
)

# ============================================================
# PONTO 31 - ERROS DE Qg NOS 10 PIORES CENÁRIOS
# ============================================================

top10_qg_real = (
    qg_total_stress
    .sort_values("total_Qg_error", ascending=False)
    .head(10)
    .copy()
)

top10_ids = top10_qg_real["scenario"].tolist()

df_top10_qg = df[
    df["scenario"].isin(top10_ids) &
    (df["Qg_target"] != 0)
].copy()

qg_top10_bus = (
    df_top10_qg
    .groupby("bus")
    .agg(
        N=("error_Qg", "size"),
        MAE_Qg=("error_Qg", "mean"),
        P95_Qg=("error_Qg", lambda x: x.quantile(0.95)),
        MAX_Qg=("error_Qg", "max"),
        mean_signed_Qg=("signed_error_Qg", "mean")
    )
    .sort_values("MAE_Qg", ascending=False)
)

print("\n" + "=" * 70)
print("ERROS DE Qg NOS 10 PIORES CENÁRIOS")
print("=" * 70)

print(qg_top10_bus)

# ============================================================
# PONTO 32 - Qg DOS 10 PIORES CENÁRIOS
# ============================================================

qg_detail_top10 = df_top10_qg[
    [
        "scenario",
        "bus",
        "Qg_target",
        "Qg_pred",
        "signed_error_Qg",
        "error_Qg"
    ]
].sort_values(["scenario", "bus"])

print("\n" + "=" * 70)
print("DETALHE DE Qg NOS 10 PIORES CENÁRIOS")
print("=" * 70)

print(qg_detail_top10.to_string(index=False))

# ============================================================
# PONTO 33 - Qg DO BUS 4 VS STRESS DE TENSÃO
# ============================================================

bus4 = df[df["bus"] == 4].copy()

bus4_scenario = (
    bus4.groupby("scenario")
    .agg(
        error_Qg=("error_Qg", "mean"),
        signed_error_Qg=("signed_error_Qg", "mean")
    )
    .join(
        scenario_analysis[["Vm_min", "stress_voltage"]]
    )
)

print("\n" + "=" * 70)
print("BUS 4 - ERRO DE Qg VS Vm_min")
print("=" * 70)

print(
    bus4_scenario.groupby("stress_voltage")["error_Qg"]
    .agg(["count", "mean", "median", "max"])
)

# ============================================================
# PONTO 34 - CASOS EXTREMOS DO BUS 4
# ============================================================

top_bus4 = (
    bus4_scenario
    .sort_values("error_Qg", ascending=False)
    .head(10)
)

print("\n" + "=" * 70)
print("TOP 10 ERROS DE Qg - BUS 4")
print("=" * 70)

print(top_bus4)

# ============================================================
# PONTO 35 - OUTROS GERADORES NOS CASOS EXTREMOS DO BUS 4
# ============================================================

top_bus4_ids = top_bus4.index.tolist()

extreme_bus4 = df[
    df["scenario"].isin(top_bus4_ids) &
    (df["Qg_target"] != 0)
].copy()

extreme_bus4_analysis = (
    extreme_bus4
    .groupby("bus")
    .agg(
        MAE_Qg=("error_Qg", "mean"),
        mean_signed_Qg=("signed_error_Qg", "mean"),
        MAX_Qg=("error_Qg", "max")
    )
    .sort_values("MAE_Qg", ascending=False)
)

print("\n" + "=" * 70)
print("OUTROS GERADORES NOS 10 CASOS EXTREMOS DO BUS 4")
print("=" * 70)

print(extreme_bus4_analysis)

# ============================================================
# PONTO 36 - COMPENSAÇÃO NOS 10 CASOS EXTREMOS DO BUS 4
# ============================================================

comp_extreme_bus4 = (
    extreme_bus4
    .groupby("scenario")
    .agg(
        individual_Qg_error=("error_Qg", "sum"),
        total_Qg_signed=("signed_error_Qg", "sum")
    )
)

comp_extreme_bus4["total_Qg_error"] = (
    comp_extreme_bus4["total_Qg_signed"].abs()
)

comp_extreme_bus4["compensation_%"] = (
    1 -
    comp_extreme_bus4["total_Qg_error"] /
    comp_extreme_bus4["individual_Qg_error"]
) * 100

print("\n" + "=" * 70)
print("COMPENSAÇÃO NOS 10 CASOS EXTREMOS DO BUS 4")
print("=" * 70)

print(comp_extreme_bus4)

# ============================================================
# PONTO 37 - COMPARAÇÃO DA COMPENSAÇÃO Qg
# ============================================================

print("\n" + "=" * 70)
print("COMPARAÇÃO DA COMPENSAÇÃO Qg")
print("=" * 70)

print("\n10 casos extremos do Bus 4:")
print(
    f"  Média:   {comp_extreme_bus4['compensation_%'].mean():.2f}%"
)
print(
    f"  Mediana: {comp_extreme_bus4['compensation_%'].median():.2f}%"
)

print("\nTodos os cenários:")
print(
    f"  Média:   {compensation_qg.mean():.2f}%"
)
print(
    f"  Mediana: {compensation_qg.median():.2f}%"
)


## Análise do Active RES


# ==========================================
# PONTO 38 - Estatísticas globais active res.
# ==========================================

res_active = df["active res. (MW)"].abs()

print("\n===== ACTIVE RES. (MW) =====")

print(f"MAE  : {res_active.mean():.8f} MW")
print(f"RMSE : {np.sqrt(np.mean(res_active**2)):.8f} MW")
print(f"P95  : {res_active.quantile(0.95):.8f} MW")
print(f"P99  : {res_active.quantile(0.99):.8f} MW")
print(f"MAX  : {res_active.max():.8f} MW")

# ==========================================
# PONTO 39 - Active residual por bus
# ==========================================

active_bus_analysis = (
    df.groupby("bus")
    .agg(
        N=("active res. (MW)", "size"),
        MAE_active=("active res. (MW)", lambda x: np.abs(x).mean()),
        RMSE_active=("active res. (MW)", lambda x: np.sqrt(np.mean(x**2))),
        P95_active=("active res. (MW)", lambda x: np.abs(x).quantile(0.95)),
        P99_active=("active res. (MW)", lambda x: np.abs(x).quantile(0.99)),
        MAX_active=("active res. (MW)", lambda x: np.abs(x).max())
    )
    .sort_values("MAE_active", ascending=False)
)

print("\n===== ACTIVE RES. POR BUS =====")
print(active_bus_analysis)

# ==========================================
# PONTO 40 - Active residual por cenário
# ==========================================

active_scenario = (
    df.groupby("scenario")
    .agg(
        Pd_total=("Pd", "sum"),
        Qd_total=("Qd", "sum"),
        Vm_min=("Vm_target", "min"),
        MAE_active=("active res. (MW)", lambda x: np.abs(x).mean()),
        MAX_active=("active res. (MW)", lambda x: np.abs(x).max())
    )
)

print("\n===== CORRELAÇÕES ACTIVE RES. =====")

print(
    active_scenario[
        ["MAE_active", "MAX_active", "Pd_total", "Qd_total", "Vm_min"]
    ].corr()
)

# ==========================================
# PONTO 41 - Active residual por stress
# ==========================================

active_scenario["stress_voltage"] = pd.cut(
    active_scenario["Vm_min"],
    bins=[-np.inf, 0.96, 0.98, np.inf],
    labels=["Stress alto", "Stress médio", "Stress baixo"]
)

active_stress = (
    active_scenario
    .groupby("stress_voltage", observed=False)
    .agg(
        N=("MAE_active", "size"),
        MAE_active=("MAE_active", "mean"),
        P95_active=("MAE_active", lambda x: x.quantile(0.95)),
        P99_active=("MAE_active", lambda x: x.quantile(0.99)),
        MAX_active=("MAX_active", "max")
    )
)

print("\n===== ACTIVE RES. POR STRESS =====")
print(active_stress)

# ==========================================
# PONTO 42 - Piores cenários de active res.
# ==========================================

top10_active = (
    active_scenario
    .sort_values("MAX_active", ascending=False)
    .head(10)
    .copy()
)

print("\n===== TOP 10 CENÁRIOS - ACTIVE RES. =====")
print(
    top10_active[
        [
            "MAX_active",
            "MAE_active",
            "Pd_total",
            "Qd_total",
            "Vm_min",
            "stress_voltage"
        ]
    ]
)

# ==========================================
# PONTO 43 - Bus responsável pelo MAX active
# ==========================================

top_scenarios_ids = top10_active.index

top_active_rows = (
    df[df["scenario"].isin(top_scenarios_ids)]
    .copy()
)

top_active_rows["abs_active_res"] = (
    top_active_rows["active res. (MW)"].abs()
)

top_active_rows = top_active_rows.loc[
    top_active_rows.groupby("scenario")["abs_active_res"].idxmax()
]

top_active_rows = top_active_rows.sort_values(
    "abs_active_res",
    ascending=False
)

print("\n===== BUS RESPONSÁVEL PELO MAX ACTIVE =====")

print(
    top_active_rows[
        [
            "scenario",
            "bus",
            "active res. (MW)",
            "abs_active_res",
            "Vm_target",
            "Pd",
            "Qd"
        ]
    ].to_string(index=False)
)

# ==========================================
# PONTO 44 - Active residual por bus e stress
# ==========================================

df["abs_active_res"] = df["active res. (MW)"].abs()

df["stress_voltage"] = pd.cut(
    df["Vm_target"],
    bins=[-np.inf, 0.96, 0.98, np.inf],
    labels=["Stress alto", "Stress médio", "Stress baixo"]
)

active_bus_stress = (
    df.groupby(["stress_voltage", "bus"], observed=False)
    .agg(
        N=("abs_active_res", "size"),
        MAE_active=("abs_active_res", "mean"),
        P95_active=("abs_active_res", lambda x: x.quantile(0.95)),
        P99_active=("abs_active_res", lambda x: x.quantile(0.99)),
        MAX_active=("abs_active_res", "max")
    )
    .reset_index()
)

print("\n===== ACTIVE RES. POR BUS E STRESS =====")
print(active_bus_stress.to_string(index=False))

# ==========================================
# PONTO 45 - Buses mais problemáticos
# ==========================================

buses_interest = [29, 28, 5]

active_focus = (
    active_bus_stress[
        active_bus_stress["bus"].isin(buses_interest)
    ]
    .sort_values(["bus", "stress_voltage"])
)

print("\n===== BUSES 29, 28 E 5 - ACTIVE RES. =====")
print(active_focus.to_string(index=False))

# ==========================================
# PONTO 46 - Active residual vs erro de Pg
# ==========================================

active_pg_analysis = (
    df.groupby("scenario")
    .agg(
        MAE_active=("active res. (MW)", lambda x: np.abs(x).mean()),
        MAX_active=("active res. (MW)", lambda x: np.abs(x).max()),
        MAE_Pg=("error_Pg", "mean"),
        Pd_total=("Pd", "sum"),
        Qd_total=("Qd", "sum"),
        Vm_min=("Vm_target", "min")
    )
)

print("\n===== CORRELAÇÃO ACTIVE RES. vs Pg =====")

print(
    active_pg_analysis[
        ["MAE_active", "MAX_active", "MAE_Pg", "Pd_total", "Qd_total", "Vm_min"]
    ].corr()
)

# ==========================================
# PONTO 47 - Pg error vs Active residual
# ==========================================

top_pg_active = (
    active_pg_analysis
    .sort_values(["MAE_Pg", "MAE_active"], ascending=False)
    .head(10)
)

print("\n===== TOP 10 CENÁRIOS - Pg + ACTIVE RES. =====")

print(
    top_pg_active[
        [
            "MAE_Pg",
            "MAE_active",
            "MAX_active",
            "Pd_total",
            "Qd_total",
            "Vm_min"
        ]
    ].to_string()
)

# ==========================================
# PONTO 48 - Bus responsável pelo active res.
# ==========================================

top_pg_scenarios = top_pg_active.index

top_pg_active_rows = df[
    df["scenario"].isin(top_pg_scenarios)
].copy()

top_pg_active_rows["abs_active_res"] = (
    top_pg_active_rows["active res. (MW)"].abs()
)

top_pg_active_rows = top_pg_active_rows.loc[
    top_pg_active_rows.groupby("scenario")["abs_active_res"].idxmax()
]

top_pg_active_rows = top_pg_active_rows.sort_values(
    "abs_active_res",
    ascending=False
)

print("\n===== BUS COM MAIOR ACTIVE RES. NOS TOP 10 Pg =====")

print(
    top_pg_active_rows[
        [
            "scenario",
            "bus",
            "active res. (MW)",
            "abs_active_res",
            "Vm_target",
            "Pd",
            "Qd"
        ]
    ].to_string(index=False)
)

# ==========================================
# PONTO 49 - Active residual vs Vm
# ==========================================

buses_interest = [28, 29, 5]

active_vm = (
    df[df["bus"].isin(buses_interest)]
    .groupby("bus")
    .apply(
        lambda g: pd.Series({
            "corr_active_Vm": g["abs_active_res"].corr(g["Vm_target"]),
            "MAE_active": g["abs_active_res"].mean(),
            "MAX_active": g["abs_active_res"].max()
        }),
        include_groups=False
    )
)

print("\n===== ACTIVE RES. vs Vm_target =====")
print(active_vm)

# ==========================================
# PONTO 50 - Resumo final buses 5, 28 e 29
# ==========================================

active_final = (
    df[df["bus"].isin([5, 28, 29])]
    .groupby("bus")
    .agg(
        N=("abs_active_res", "size"),
        MAE=("abs_active_res", "mean"),
        RMSE=("abs_active_res", lambda x: np.sqrt(np.mean(x**2))),
        P95=("abs_active_res", lambda x: x.quantile(0.95)),
        P99=("abs_active_res", lambda x: x.quantile(0.99)),
        MAX=("abs_active_res", "max")
    )
)

print("\n===== RESUMO FINAL - ACTIVE RES. =====")
print(active_final.to_string())

# ==========================================
# PONTO 51 - Stress por cenário
# ==========================================

scenario_stress = (
    scenario_analysis[["Vm_min"]]
    .copy()
)

scenario_stress["stress_voltage"] = pd.cut(
    scenario_stress["Vm_min"],
    bins=[-np.inf, 0.96, 0.98, np.inf],
    labels=["Stress alto", "Stress médio", "Stress baixo"]
)

df["stress_scenario"] = df["scenario"].map(
    scenario_stress["stress_voltage"]
)

print("\n===== DISTRIBUIÇÃO DO STRESS POR CENÁRIO =====")
print(
    df.groupby(["stress_scenario", "bus"], observed=False)
    .size()
    .unstack(fill_value=0)
)

# ==========================================
# PONTO 52 - Active res. por bus e stress
# usando stress definido ao nível do cenário
# ==========================================

active_bus_stress_correct = (
    df[df["bus"].isin([5, 28, 29])]
    .groupby(["stress_scenario", "bus"], observed=False)
    .agg(
        N=("abs_active_res", "size"),
        MAE_active=("abs_active_res", "mean"),
        RMSE_active=("abs_active_res", lambda x: np.sqrt(np.mean(x**2))),
        P95_active=("abs_active_res", lambda x: x.quantile(0.95)),
        P99_active=("abs_active_res", lambda x: x.quantile(0.99)),
        MAX_active=("abs_active_res", "max")
    )
    .reset_index()
)

print("\n===== ACTIVE RES. - STRESS POR CENÁRIO =====")

print(
    active_bus_stress_correct.to_string(index=False)
)

# ==========================================
# PONTO 53 - Degradação por stress em todos os buses
# ==========================================

active_all_bus_stress = (
    df.groupby(["stress_scenario", "bus"], observed=False)
    .agg(
        MAE_active=("abs_active_res", "mean")
    )
    .reset_index()
)

active_pivot = active_all_bus_stress.pivot(
    index="bus",
    columns="stress_scenario",
    values="MAE_active"
)

active_pivot["ratio_alto_baixo"] = (
    active_pivot["Stress alto"] /
    active_pivot["Stress baixo"]
)

active_pivot = active_pivot.sort_values(
    "ratio_alto_baixo",
    ascending=False
)

print("\n===== DEGRADAÇÃO DO ACTIVE RES. POR STRESS =====")

print(active_pivot.to_string())

# ==========================================
# PONTO 54 - P95/P99: Stress alto vs baixo
# ==========================================

active_stress_compare = (
    df.groupby(["stress_scenario", "bus"], observed=False)
    .agg(
        MAE=("abs_active_res", "mean"),
        P95=("abs_active_res", lambda x: x.quantile(0.95)),
        P99=("abs_active_res", lambda x: x.quantile(0.99)),
        MAX=("abs_active_res", "max")
    )
    .reset_index()
)

active_compare_pivot = active_stress_compare.pivot(
    index="bus",
    columns="stress_scenario"
)

print("\n===== P95/P99 - STRESS ALTO VS BAIXO =====")

print(active_compare_pivot.to_string())

# ==========================================
# PONTO 55 - Contribuição dos buses
# ==========================================

active_contribution = (
    df.groupby(["stress_scenario", "bus"], observed=False)
    .agg(
        total_active_error=("abs_active_res", "sum"),
        MAE_active=("abs_active_res", "mean")
    )
    .reset_index()
)

for stress in ["Stress alto", "Stress médio", "Stress baixo"]:
    
    temp = (
        active_contribution[
            active_contribution["stress_scenario"] == stress
        ]
        .sort_values("total_active_error", ascending=False)
        .head(10)
    )

    print(f"\n===== TOP 10 BUSES - {stress} =====")
    print(temp.to_string(index=False))

# ==========================================
# PONTO 56 - Active residual vs PBE
# ==========================================

active_pbe_analysis = (
    df.groupby("scenario")
    .agg(
        MAE_active=("abs_active_res", "mean"),
        MAX_active=("abs_active_res", "max"),
        PBE=("PBE", "mean"),
        Pd_total=("Pd", "sum"),
        Qd_total=("Qd", "sum"),
        Vm_min=("Vm_target", "min")
    )
)

print("\n===== CORRELAÇÕES: ACTIVE RES. VS PBE =====")

print(
    active_pbe_analysis[
        ["MAE_active", "MAX_active", "PBE",
         "Pd_total", "Qd_total", "Vm_min"]
    ].corr()
)

# ==========================================
# PONTO 57 - PBE vs Active Residual
# ==========================================

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(
    active_pbe_analysis["PBE"],
    active_pbe_analysis["MAE_active"],
    alpha=0.3,
    s=10
)

plt.xlabel("PBE")
plt.ylabel("MAE do active res. (MW)")
plt.title("Relação entre PBE e Active Residual")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

from scipy.stats import linregress

x = active_pbe_analysis["PBE"]
y = active_pbe_analysis["MAE_active"]

reg = linregress(x, y)

print("\n===== REGRESSÃO LINEAR: PBE vs ACTIVE RES. =====")
print(f"Declive:      {reg.slope:.6f}")
print(f"Intercepto:   {reg.intercept:.6f}")
print(f"Correlação r: {reg.rvalue:.6f}")
print(f"R²:           {reg.rvalue**2:.6f}")
print(f"p-value:      {reg.pvalue:.3e}")



# ============================================================
# PONTO 58 - VERIFICACAO DOS LIMITES OPERACIONAIS DE Qg
# ============================================================

print("\n" + "=" * 75)
print("VERIFICACAO DOS LIMITES DE POTENCIA REATIVA - Qg")
print("=" * 75)

# Verificar se existem as colunas necessarias
required_columns = [
    "scenario", "bus", "PV", "REF",
    "Qg_target", "Qg_pred", "Qg_min", "Qg_max"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Faltam colunas necessarias no dataset: {missing_columns}"
    )

# Analisar apenas buses identificados como PV ou REF
# e linhas com limites validos.
qg_limits = df.copy()

qg_limits["Qg_min"] = pd.to_numeric(
    qg_limits["Qg_min"], errors="coerce"
)
qg_limits["Qg_max"] = pd.to_numeric(
    qg_limits["Qg_max"], errors="coerce"
)

generator_mask = (
    (qg_limits["PV"] == 1) |
    (qg_limits["REF"] == 1)
)

valid_limits = (
    qg_limits["Qg_min"].notna() &
    qg_limits["Qg_max"].notna() &
    np.isfinite(qg_limits["Qg_min"]) &
    np.isfinite(qg_limits["Qg_max"]) &
    (qg_limits["Qg_min"] <= qg_limits["Qg_max"])
)

qg_limits = qg_limits[generator_mask & valid_limits].copy()

if qg_limits.empty:
    raise ValueError(
        "Nao foram encontradas linhas de geradores com limites Qg validos."
    )

# Tolerancia numerica para evitar classificar pequenas diferencas
# de arredondamento como violacoes.
TOL_QG = 1e-6  # MVAr

# ------------------------------------------------------------
# 1. Verificar violacoes das previsoes
# ------------------------------------------------------------

qg_limits["violation_pred_below"] = np.maximum(
    qg_limits["Qg_min"] - qg_limits["Qg_pred"], 0
)

qg_limits["violation_pred_above"] = np.maximum(
    qg_limits["Qg_pred"] - qg_limits["Qg_max"], 0
)

qg_limits["violation_pred_MVAr"] = (
    qg_limits["violation_pred_below"] +
    qg_limits["violation_pred_above"]
)

qg_limits["pred_out_of_limits"] = (
    qg_limits["violation_pred_MVAr"] > TOL_QG
)

# ------------------------------------------------------------
# 2. Verificar tambem os targets
# ------------------------------------------------------------

qg_limits["violation_target_below"] = np.maximum(
    qg_limits["Qg_min"] - qg_limits["Qg_target"], 0
)

qg_limits["violation_target_above"] = np.maximum(
    qg_limits["Qg_target"] - qg_limits["Qg_max"], 0
)

qg_limits["violation_target_MVAr"] = (
    qg_limits["violation_target_below"] +
    qg_limits["violation_target_above"]
)

qg_limits["target_out_of_limits"] = (
    qg_limits["violation_target_MVAr"] > TOL_QG
)

# ------------------------------------------------------------
# 3. Resumo global
# ------------------------------------------------------------

n_rows = len(qg_limits)
n_pred_violations = int(qg_limits["pred_out_of_limits"].sum())
n_target_violations = int(qg_limits["target_out_of_limits"].sum())

print(f"\nLinhas de geradores com limites validos: {n_rows}")

print(
    f"Previsoes fora dos limites: {n_pred_violations} "
    f"({100 * n_pred_violations / n_rows:.4f}%)"
)

print(
    f"Targets fora dos limites:   {n_target_violations} "
    f"({100 * n_target_violations / n_rows:.4f}%)"
)

print(
    "\nViolacao maxima das previsoes: "
    f"{qg_limits['violation_pred_MVAr'].max():.6f} MVAr"
)

print(
    "Violacao media, considerando apenas previsoes violadoras: "
    f"{qg_limits.loc[qg_limits['pred_out_of_limits'], 'violation_pred_MVAr'].mean():.6f} MVAr"
    if n_pred_violations > 0
    else "Nao existem previsoes fora dos limites."
)

# ------------------------------------------------------------
# 4. Resumo por bus
# ------------------------------------------------------------

qg_limits_by_bus = (
    qg_limits.groupby("bus")
    .agg(
        N=("scenario", "size"),
        pred_violations=("pred_out_of_limits", "sum"),
        target_violations=("target_out_of_limits", "sum"),
        pred_violation_rate=("pred_out_of_limits", "mean"),
        target_violation_rate=("target_out_of_limits", "mean"),
        mean_pred_violation_MVAr=("violation_pred_MVAr", "mean"),
        max_pred_violation_MVAr=("violation_pred_MVAr", "max"),
        mean_Qg_error=("error_Qg", "mean"),
        Qg_min=("Qg_min", "min"),
        Qg_max=("Qg_max", "max")
    )
)

qg_limits_by_bus["pred_violation_rate"] *= 100
qg_limits_by_bus["target_violation_rate"] *= 100

print("\n===== VIOLACOES DOS LIMITES Qg POR BUS =====")
print(qg_limits_by_bus.round(6).to_string())

# ------------------------------------------------------------
# 5. Top 20 previsoes com maiores violacoes
# ------------------------------------------------------------

violations_top = (
    qg_limits[qg_limits["pred_out_of_limits"]]
    .sort_values("violation_pred_MVAr", ascending=False)
    [[
        "scenario", "bus", "Qg_target", "Qg_pred",
        "Qg_min", "Qg_max", "violation_pred_below",
        "violation_pred_above", "violation_pred_MVAr",
        "violation_target_MVAr", "Vm_target"
    ]]
    .head(20)
)

print("\n===== TOP 20 VIOLACOES DE Qg =====")

if violations_top.empty:
    print("Nenhuma previsao violou os limites de Qg.")
else:
    print(violations_top.to_string(index=False))

# ------------------------------------------------------------
# 6. Guardar resultados
# ------------------------------------------------------------

OUTPUT.mkdir(parents=True, exist_ok=True)

summary_file = OUTPUT / "case30_Qg_limit_violations_by_bus.csv"
violations_file = OUTPUT / "case30_Qg_limit_violations_top.csv"

qg_limits_by_bus.to_csv(summary_file)
violations_top.to_csv(violations_file, index=False)

print("\n===== FICHEIROS GUARDADOS =====")
print(summary_file)
print(violations_file)



# ============================================================
# 59. DISTRIBUICAO DAS VIOLACOES DOS LIMITES DE Qg
# ============================================================

print("\n" + "=" * 75)
print("DISTRIBUICAO DAS VIOLACOES DOS LIMITES DE Qg")
print("=" * 75)

# Buses com limites de Qg validos na analise anterior
buses_qg = [0, 1, 4, 7, 10, 12]

# Selecionar apenas os 6 buses e registos com dados validos
df_qg = df.loc[
    df["bus"].isin(buses_qg)
    & df["Qg_min"].notna()
    & df["Qg_max"].notna()
    & df["Qg_pred"].notna()
].copy()

# Calcular a violacao dos limites de Qg, em MVAr
# Valor positivo: excede um dos limites
# Zero: previsao dentro dos limites
df_qg["violacao_Qg_MVAr"] = np.maximum(
    df_qg["Qg_min"] - df_qg["Qg_pred"],
    df_qg["Qg_pred"] - df_qg["Qg_max"]
).clip(lower=0)

# Filtrar previsoes efetivamente fora dos limites
TOL_QG = 1e-6

violacoes = df_qg.loc[
    df_qg["violacao_Qg_MVAr"] > TOL_QG
].copy()

# Resumo geral
total = len(df_qg)
total_violacoes = len(violacoes)

print(f"\nBuses analisados: {buses_qg}")
print(f"Total de previsoes analisadas: {total}")
print(
    f"Previsoes fora dos limites: {total_violacoes} "
    f"({100 * total_violacoes / total if total else 0:.4f}%)"
)

# Distribuicao por limiar de violacao
limiares = [0.01, 0.1, 0.5]
resultados = []

for limite in limiares:

    # Numero de violacoes superiores ao limiar
    n = int(
        (violacoes["violacao_Qg_MVAr"] > limite).sum()
    )

    percentagem_total = (
        100 * n / total if total else 0
    )

    percentagem_violacoes = (
        100 * n / total_violacoes
        if total_violacoes else 0
    )

    resultados.append({
        "limiar_MVAr": limite,
        "numero_violacoes": n,
        "percentagem_total": percentagem_total,
        "percentagem_violacoes": percentagem_violacoes
    })

    print(f"\nViolacoes > {limite:.2f} MVAr:")
    print(f"  Numero: {n}")
    print(f"  Percentagem do total: {percentagem_total:.4f}%")
    print(
        f"  Percentagem das violacoes: "
        f"{percentagem_violacoes:.2f}%"
    )

# Guardar resultados em CSV
df_distribuicao = pd.DataFrame(resultados)

ficheiro_saida = (
    OUTPUT / "case30_Qg_violation_distribution.csv"
)

ficheiro_saida.parent.mkdir(parents=True, exist_ok=True)
df_distribuicao.to_csv(ficheiro_saida, index=False)

print("\n===== FICHEIRO GUARDADO =====")
print(ficheiro_saida)



# ============================================================
# 60. RELACAO ENTRE VIOLACOES DE Qg E TENSAO Vm
# ============================================================

print("\n" + "=" * 75)
print("RELACAO ENTRE VIOLACOES DE Qg E TENSAO Vm")
print("=" * 75)

# Buses com limites de Qg analisados anteriormente
buses_qg = [0, 1, 4, 7, 10, 12]

# Selecionar os registos relevantes
df_vm_qg = df.loc[
    df["bus"].isin(buses_qg)
    & df["Qg_min"].notna()
    & df["Qg_max"].notna()
    & df["Qg_pred"].notna()
    & df["Vm_target"].notna()
].copy()

# Calcular a violacao dos limites de Qg
df_vm_qg["violacao_Qg_MVAr"] = np.maximum(
    df_vm_qg["Qg_min"] - df_vm_qg["Qg_pred"],
    df_vm_qg["Qg_pred"] - df_vm_qg["Qg_max"]
).clip(lower=0)

TOL_QG = 1e-6

df_vm_qg["Qg_violacao"] = (
    df_vm_qg["violacao_Qg_MVAr"] > TOL_QG
)

# Distancia de Vm_target aos limites de tensao de referencia
# Usar 0.94 e 1.06 p.u. como intervalo de referencia
VM_MIN = 0.94
VM_MAX = 1.06

df_vm_qg["distancia_limite_Vm"] = np.minimum(
    df_vm_qg["Vm_target"] - VM_MIN,
    VM_MAX - df_vm_qg["Vm_target"]
)

# Identificar valores de tensao proximos dos limites
MARGEM_VM = 0.01

df_vm_qg["Vm_proximo_limite"] = (
    (df_vm_qg["Vm_target"] <= VM_MIN + MARGEM_VM)
    | (df_vm_qg["Vm_target"] >= VM_MAX - MARGEM_VM)
)

# ------------------------------------------------------------
# 1. Comparar casos com e sem violacao de Qg
# ------------------------------------------------------------

resumo = df_vm_qg.groupby("Qg_violacao").agg(
    numero_registos=("Qg_violacao", "size"),
    Vm_media=("Vm_target", "mean"),
    Vm_minimo=("Vm_target", "min"),
    Vm_maximo=("Vm_target", "max"),
    Vm_desvio_padrao=("Vm_target", "std"),
    percentagem_proxima_limite=(
        "Vm_proximo_limite",
        lambda x: 100 * x.mean()
    ),
    violacao_media_MVAr=("violacao_Qg_MVAr", "mean"),
    violacao_maxima_MVAr=("violacao_Qg_MVAr", "max")
)

print("\n===== COMPARACAO: COM E SEM VIOLACAO DE Qg =====")
print(resumo.to_string())

# ------------------------------------------------------------
# 2. Agrupar por intervalos de tensao
# ------------------------------------------------------------

bins_vm = [
    -np.inf,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
    1.00,
    1.01,
    1.02,
    1.03,
    1.04,
    1.05,
    1.06,
    np.inf
]

labels_vm = [
    "<0.94",
    "0.94-0.95",
    "0.95-0.96",
    "0.96-0.97",
    "0.97-0.98",
    "0.98-0.99",
    "0.99-1.00",
    "1.00-1.01",
    "1.01-1.02",
    "1.02-1.03",
    "1.03-1.04",
    "1.04-1.05",
    "1.05-1.06",
    ">1.06"
]

df_vm_qg["intervalo_Vm"] = pd.cut(
    df_vm_qg["Vm_target"],
    bins=bins_vm,
    labels=labels_vm,
    right=False
)

resumo_vm = df_vm_qg.groupby(
    "intervalo_Vm",
    observed=False
).agg(
    numero_registos=("Qg_violacao", "size"),
    numero_violacoes=("Qg_violacao", "sum"),
    violacao_media_MVAr=("violacao_Qg_MVAr", "mean")
)

resumo_vm["taxa_violacao_percentagem"] = (
    100 * resumo_vm["numero_violacoes"]
    / resumo_vm["numero_registos"].replace(0, np.nan)
)

print("\n===== VIOLACOES POR INTERVALO DE Vm_target =====")
print(resumo_vm.to_string())

# ------------------------------------------------------------
# 3. Comparar tensao proxima dos limites vs restante
# ------------------------------------------------------------

resumo_proximidade = df_vm_qg.groupby(
    "Vm_proximo_limite"
).agg(
    numero_registos=("Qg_violacao", "size"),
    numero_violacoes=("Qg_violacao", "sum"),
    violacao_media_MVAr=("violacao_Qg_MVAr", "mean")
)

resumo_proximidade["taxa_violacao_percentagem"] = (
    100 * resumo_proximidade["numero_violacoes"]
    / resumo_proximidade["numero_registos"]
)

print("\n===== PROXIMIDADE AOS LIMITES DE TENSAO =====")
print("Proximo limite: Vm <= 0.95 ou Vm >= 1.05 p.u.")
print(resumo_proximidade.to_string())

# ------------------------------------------------------------
# 4. Guardar resultados
# ------------------------------------------------------------

ficheiro_resumo = (
    OUTPUT / "case30_Qg_vs_Vm_summary.csv"
)

ficheiro_intervalos = (
    OUTPUT / "case30_Qg_vs_Vm_intervals.csv"
)

ficheiro_proximidade = (
    OUTPUT / "case30_Qg_vs_Vm_proximity.csv"
)

ficheiro_resumo.parent.mkdir(
    parents=True,
    exist_ok=True
)

resumo.to_csv(ficheiro_resumo)
resumo_vm.to_csv(ficheiro_intervalos)
resumo_proximidade.to_csv(ficheiro_proximidade)

print("\n===== FICHEIROS GUARDADOS =====")
print(ficheiro_resumo)
print(ficheiro_intervalos)
print(ficheiro_proximidade)


# ============================================================
# 61. VIOLACOES DE Qg POR BUS E PROXIMIDADE DA TENSAO
# ============================================================

print("\n" + "=" * 75)
print("VIOLACOES DE Qg POR BUS E PROXIMIDADE DA TENSAO")
print("=" * 75)

buses_qg = [0, 1, 4, 7, 10, 12]

# Reutilizar df_vm_qg criado no bloco 60
# Se nao existir, criar novamente
if "df_vm_qg" not in globals():
    df_vm_qg = df.loc[
        df["bus"].isin(buses_qg)
        & df["Qg_min"].notna()
        & df["Qg_max"].notna()
        & df["Qg_pred"].notna()
        & df["Vm_target"].notna()
    ].copy()

    df_vm_qg["violacao_Qg_MVAr"] = np.maximum(
        df_vm_qg["Qg_min"] - df_vm_qg["Qg_pred"],
        df_vm_qg["Qg_pred"] - df_vm_qg["Qg_max"]
    ).clip(lower=0)

    df_vm_qg["Qg_violacao"] = (
        df_vm_qg["violacao_Qg_MVAr"] > 1e-6
    )

    df_vm_qg["Vm_proximo_limite"] = (
        (df_vm_qg["Vm_target"] <= 0.95)
        | (df_vm_qg["Vm_target"] >= 1.05)
    )

# Construir resumo por bus e grupo de tensao
resumo_bus_vm = df_vm_qg.groupby(
    ["bus", "Vm_proximo_limite"],
    observed=True
).agg(
    numero_registos=("Qg_violacao", "size"),
    numero_violacoes=("Qg_violacao", "sum"),
    violacao_media_MVAr=("violacao_Qg_MVAr", "mean")
).reset_index()

resumo_bus_vm["taxa_violacao_percentagem"] = (
    100
    * resumo_bus_vm["numero_violacoes"]
    / resumo_bus_vm["numero_registos"]
)

# Separar grupos para comparar as taxas
taxas = resumo_bus_vm.pivot(
    index="bus",
    columns="Vm_proximo_limite",
    values="taxa_violacao_percentagem"
).rename(columns={
    False: "taxa_tensao_afastada_pct",
    True: "taxa_tensao_proxima_pct"
})

# Garantir que as duas colunas existem mesmo se faltar um grupo
for coluna in [
    "taxa_tensao_afastada_pct",
    "taxa_tensao_proxima_pct"
]:
    if coluna not in taxas.columns:
        taxas[coluna] = np.nan

taxas["diferenca_pontos_percentuais"] = (
    taxas["taxa_tensao_afastada_pct"]
    - taxas["taxa_tensao_proxima_pct"]
)

taxas = taxas.reset_index()

print("\n===== CONTAGENS E TAXAS POR BUS =====")
print(resumo_bus_vm.to_string(index=False))

print("\n===== COMPARACAO DAS TAXAS POR BUS =====")
print(taxas.to_string(index=False))

# Guardar os resultados
ficheiro_detalhe = (
    OUTPUT / "case30_Qg_vs_Vm_by_bus_detail.csv"
)

ficheiro_comparacao = (
    OUTPUT / "case30_Qg_vs_Vm_by_bus_comparison.csv"
)

ficheiro_detalhe.parent.mkdir(
    parents=True,
    exist_ok=True
)

resumo_bus_vm.to_csv(ficheiro_detalhe, index=False)
taxas.to_csv(ficheiro_comparacao, index=False)

print("\n===== FICHEIROS GUARDADOS =====")
print(ficheiro_detalhe)
print(ficheiro_comparacao)


# # ==========================================
# # PONTO 59 - PBE vs Active Residual por stress
# # ==========================================

# from scipy.stats import linregress
# import numpy as np
# import pandas as pd

# # Calcular métricas por cenário
# pbe_stress_analysis = (
#     df.groupby(["scenario", "stress_scenario"], observed=True)
#     .agg(
#         MAE_active=("abs_active_res", "mean"),
#         PBE=("PBE", "mean")
#     )
#     .reset_index()
# )

# print("\n===== PBE VS ACTIVE RES. POR STRESS =====")

# for stress in ["Stress alto", "Stress médio", "Stress baixo"]:

#     subset = pbe_stress_analysis[
#         pbe_stress_analysis["stress_scenario"] == stress
#     ]

#     reg = linregress(
#         subset["PBE"],
#         subset["MAE_active"]
#     )

#     print(f"\n--- {stress} ---")
#     print(f"N cenários:   {len(subset)}")
#     print(f"Correlação r: {reg.rvalue:.6f}")
#     print(f"R²:           {reg.rvalue**2:.6f}")
#     print(f"Declive:      {reg.slope:.6f}")
#     print(f"Intercepto:   {reg.intercept:.6f}")



# # ==========================================
# # PONTO 60 - Resíduos da regressão linear
# # ==========================================

# import matplotlib.pyplot as plt
# from scipy.stats import linregress

# x = active_pbe_analysis["PBE"]
# y = active_pbe_analysis["MAE_active"]

# reg = linregress(x, y)

# # Valores previstos pela regressão
# y_pred = reg.intercept + reg.slope * x

# # Resíduos da regressão
# regression_residuals = y - y_pred

# print("\n===== RESÍDUOS DA REGRESSÃO PBE VS ACTIVE RES. =====")
# print(f"Resíduo médio: {regression_residuals.mean():.8f}")
# print(f"MAE da regressão: {regression_residuals.abs().mean():.8f}")
# print(f"RMSE da regressão: {(regression_residuals.pow(2).mean())**0.5:.8f}")
# print(f"Resíduo máximo absoluto: {regression_residuals.abs().max():.8f}")

# # Gráfico dos resíduos
# plt.figure(figsize=(8, 6))
# plt.scatter(y_pred, regression_residuals, alpha=0.3, s=10)
# plt.axhline(0, linestyle="--")
# plt.xlabel("MAE do active res. previsto (MW)")
# plt.ylabel("Resíduo da regressão (MW)")
# plt.title("Resíduos da regressão linear: PBE vs Active Residual")
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# # ============================================================
# # PONTO 61 - Resíduos da regressão por nível de stress
# # ============================================================

# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# # Associar o nível de stress de cada cenário
# active_pbe_analysis["stress_voltage"] = (
#     active_pbe_analysis.index.map(
#         scenario_stress["stress_voltage"]
#     )
# )

# # Guardar os resíduos da regressão
# active_pbe_analysis["regression_residual"] = regression_residuals

# active_pbe_analysis["abs_regression_residual"] = (
#     active_pbe_analysis["regression_residual"].abs()
# )

# # Estatísticas por nível de stress
# residual_stress_analysis = (
#     active_pbe_analysis
#     .groupby("stress_voltage", observed=True)
#     .agg(
#         N=("regression_residual", "count"),
#         Mean_residual=("regression_residual", "mean"),
#         MAE_residual=("abs_regression_residual", "mean"),
#         RMSE_residual=(
#             "regression_residual",
#             lambda x: np.sqrt(np.mean(x**2))
#         ),
#         Max_abs_residual=("abs_regression_residual", "max"),
#         P95_abs_residual=(
#             "abs_regression_residual",
#             lambda x: x.quantile(0.95)
#         )
#     )
# )

# print("\n===== PONTO 61 - RESÍDUOS POR STRESS =====")
# print(residual_stress_analysis.round(8))

# # Gráfico dos resíduos por nível de stress
# plt.figure(figsize=(9, 6))

# stress_order = ["Stress baixo", "Stress médio", "Stress alto"]

# data_to_plot = [
#     active_pbe_analysis.loc[
#         active_pbe_analysis["stress_voltage"] == stress,
#         "regression_residual"
#     ].dropna().values
#     for stress in stress_order
# ]

# plt.boxplot(data_to_plot, tick_labels=stress_order, showfliers=True)
# plt.axhline(0, linestyle="--")
# plt.xlabel("Nível de stress de tensão")
# plt.ylabel("Resíduo da regressão (MW)")
# plt.title("Resíduos da regressão PBE vs Active Residual por stress")
# plt.grid(True, axis="y", alpha=0.3)
# plt.tight_layout()
# plt.show()

# # ============================================================
# # PONTO 62 - Cenários com maiores resíduos da regressão
# # ============================================================

# # Criar tabela com os dados necessários
# outlier_analysis = active_pbe_analysis.copy()

# outlier_analysis["abs_regression_residual"] = (
#     outlier_analysis["regression_residual"].abs()
# )

# # Selecionar os 20 cenários com maiores desvios absolutos
# top_outliers = (
#     outlier_analysis
#     .sort_values("abs_regression_residual", ascending=False)
#     .head(20)
# )

# columns_to_show = [
#     "PBE",
#     "MAE_active",
#     "regression_residual",
#     "abs_regression_residual",
#     "stress_voltage",
#     "Vm_min",
#     "Pd_total",
#     "Qd_total"
# ]

# print("\n===== PONTO 62 - TOP 20 DESVIOS DA REGRESSÃO =====")
# print(top_outliers[columns_to_show].to_string(
#     float_format=lambda x: f"{x:.8f}"
# ))

# # Contar os níveis de stress nos 20 maiores desvios
# print("\n===== DISTRIBUIÇÃO DO STRESS NOS TOP 20 =====")
# print(top_outliers["stress_voltage"].value_counts())

# # Gráfico: PBE vs MAE observado, destacando os 20 cenários
# plt.figure(figsize=(9, 6))

# plt.scatter(
#     active_pbe_analysis["PBE"],
#     active_pbe_analysis["MAE_active"],
#     alpha=0.25,
#     s=10,
#     label="Restantes cenários"
# )

# plt.scatter(
#     top_outliers["PBE"],
#     top_outliers["MAE_active"],
#     s=35,
#     label="Top 20 maiores resíduos"
# )

# x_line = np.linspace(
#     active_pbe_analysis["PBE"].min(),
#     active_pbe_analysis["PBE"].max(),
#     200
# )

# plt.plot(
#     x_line,
#     reg.intercept + reg.slope * x_line,
#     linestyle="--",
#     label="Regressão linear"
# )

# plt.xlabel("PBE")
# plt.ylabel("MAE do active res. (MW)")
# plt.title("Cenários com maiores desvios da regressão")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# # ============================================================
# # PONTO 63 - Caracterização dos maiores desvios
# # ============================================================

# # Selecionar os 20 cenários com maiores resíduos absolutos
# top_outliers = (
#     active_pbe_analysis
#     .assign(
#         abs_regression_residual=lambda d:
#             d["regression_residual"].abs()
#     )
#     .nlargest(20, "abs_regression_residual")
#     .copy()
# )

# # Mostrar os dados relevantes
# columns_to_show = [
#     "PBE",
#     "MAE_active",
#     "regression_residual",
#     "abs_regression_residual",
#     "stress_voltage",
#     "Vm_min",
#     "Pd_total",
#     "Qd_total"
# ]

# print("\n===== PONTO 63 - TOP 20 CENÁRIOS =====")
# print(top_outliers[columns_to_show].to_string(
#     float_format=lambda x: f"{x:.8f}"
# ))

# # Estatísticas dos cenários extremos
# print("\n===== RESUMO DOS TOP 20 =====")
# print("Tensão mínima média:",
#       top_outliers["Vm_min"].mean())
# print("Tensão mínima mais baixa:",
#       top_outliers["Vm_min"].min())
# print("Tensão mínima mais alta:",
#       top_outliers["Vm_min"].max())

# print("\nDistribuição do stress:")
# print(top_outliers["stress_voltage"].value_counts())

# # ============================================================
# # PONTO 64 - Desvios da regressão vs carga total
# # ============================================================

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Garantir que temos o valor absoluto dos resíduos
# active_pbe_analysis["abs_regression_residual"] = (
#     active_pbe_analysis["regression_residual"].abs()
# )

# # Correlações globais
# print("\n===== PONTO 64 - CORRELAÇÕES GLOBAIS =====")

# correlations = active_pbe_analysis[
#     ["abs_regression_residual", "Pd_total", "Qd_total", "Vm_min"]
# ].corr(method="pearson")

# print(correlations.round(4))

# # Estatísticas por nível de stress
# stress_load_analysis = (
#     active_pbe_analysis
#     .groupby("stress_voltage", observed=True)
#     .agg(
#         N=("abs_regression_residual", "count"),
#         MAE_desvio=("abs_regression_residual", "mean"),
#         Corr_Pd=("abs_regression_residual",
#                  lambda x: x.corr(
#                      active_pbe_analysis.loc[x.index, "Pd_total"]
#                  )),
#         Corr_Qd=("abs_regression_residual",
#                  lambda x: x.corr(
#                      active_pbe_analysis.loc[x.index, "Qd_total"]
#                  )),
#         Pd_medio=("Pd_total", "mean"),
#         Qd_medio=("Qd_total", "mean")
#     )
# )

# print("\n===== CORRELAÇÕES E CARGAS POR STRESS =====")
# print(stress_load_analysis.round(5))

# # Gráficos: carga ativa e reativa vs desvio absoluto
# fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# stress_colors = {
#     "Stress baixo": "green",
#     "Stress médio": "orange",
#     "Stress alto": "red"
# }

# for stress, group in active_pbe_analysis.groupby(
#     "stress_voltage", observed=True
# ):
#     axes[0].scatter(
#         group["Pd_total"],
#         group["abs_regression_residual"],
#         alpha=0.35,
#         s=12,
#         color=stress_colors.get(str(stress), "gray"),
#         label=str(stress)
#     )

#     axes[1].scatter(
#         group["Qd_total"],
#         group["abs_regression_residual"],
#         alpha=0.35,
#         s=12,
#         color=stress_colors.get(str(stress), "gray"),
#         label=str(stress)
#     )

# axes[0].set_xlabel("Carga ativa total Pd (MW)")
# axes[0].set_ylabel("Desvio absoluto da regressão (MW)")
# axes[0].set_title("Desvio da regressão vs carga ativa")

# axes[1].set_xlabel("Carga reativa total Qd (MVar)")
# axes[1].set_ylabel("Desvio absoluto da regressão (MW)")
# axes[1].set_title("Desvio da regressão vs carga reativa")

# for ax in axes:
#     ax.grid(True, alpha=0.3)
#     ax.legend()

# plt.tight_layout()
# plt.show()

# # ============================================================
# # PONTO 65 - PBE vs desvio da regressão por stress
# # ============================================================

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Garantir que temos o desvio absoluto
# active_pbe_analysis["abs_regression_residual"] = (
#     active_pbe_analysis["regression_residual"].abs()
# )

# # Correlações e estatísticas por nível de stress
# results_pbe_stress = []

# for stress, group in active_pbe_analysis.groupby(
#     "stress_voltage", observed=True
# ):
#     corr = group["PBE"].corr(
#         group["abs_regression_residual"]
#     )

#     results_pbe_stress.append({
#         "Stress": str(stress),
#         "N": len(group),
#         "Corr_PBE_desvio": corr,
#         "PBE_medio": group["PBE"].mean(),
#         "Desvio_medio": group["abs_regression_residual"].mean(),
#         "Desvio_P95": group["abs_regression_residual"].quantile(0.95),
#         "Desvio_max": group["abs_regression_residual"].max()
#     })

# results_pbe_stress = pd.DataFrame(results_pbe_stress)

# print("\n===== PONTO 65 - PBE VS DESVIO POR STRESS =====")
# print(results_pbe_stress.round(8).to_string(index=False))

# # Gráfico por nível de stress
# fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# stress_order = ["Stress baixo", "Stress médio", "Stress alto"]
# stress_colors = {
#     "Stress baixo": "green",
#     "Stress médio": "orange",
#     "Stress alto": "red"
# }

# for ax, stress in zip(axes, stress_order):
#     group = active_pbe_analysis[
#         active_pbe_analysis["stress_voltage"] == stress
#     ]

#     ax.scatter(
#         group["PBE"],
#         group["abs_regression_residual"],
#         alpha=0.35,
#         s=12,
#         color=stress_colors[stress]
#     )

#     ax.set_title(stress)
#     ax.set_xlabel("PBE")
#     ax.grid(True, alpha=0.3)

# axes[0].set_ylabel("Desvio absoluto da regressão (MW)")

# fig.suptitle("PBE vs desvio absoluto da regressão por stress")
# plt.tight_layout()
# plt.show()

# # ============================================================
# # PONTO 66 - R² da regressão por nível de stress
# # ============================================================

# import numpy as np
# import pandas as pd
# from scipy.stats import linregress

# results_r2_stress = []

# stress_order = ["Stress baixo", "Stress médio", "Stress alto"]

# for stress in stress_order:

#     group = active_pbe_analysis[
#         active_pbe_analysis["stress_voltage"] == stress
#     ].dropna(subset=["PBE", "MAE_active"])

#     # Variáveis da regressão
#     x = group["PBE"]
#     y = group["MAE_active"]

#     # Regressão linear
#     reg_stress = linregress(x, y)

#     # Valores previstos pela reta
#     y_pred = reg_stress.intercept + reg_stress.slope * x

#     # R² calculado a partir dos valores observados e previstos
#     ss_res = np.sum((y - y_pred) ** 2)
#     ss_tot = np.sum((y - y.mean()) ** 2)
#     r2 = 1 - ss_res / ss_tot

#     results_r2_stress.append({
#         "Stress": stress,
#         "N": len(group),
#         "Slope": reg_stress.slope,
#         "Intercept": reg_stress.intercept,
#         "Correlation": reg_stress.rvalue,
#         "R2": r2,
#         "MAE_regression": np.mean(np.abs(y - y_pred)),
#         "RMSE_regression": np.sqrt(np.mean((y - y_pred) ** 2))
#     })

# results_r2_stress = pd.DataFrame(results_r2_stress)

# print("\n===== PONTO 66 - REGRESSÃO POR NÍVEL DE STRESS =====")
# print(results_r2_stress.round(8).to_string(index=False))

# print("\n===== R² EM PERCENTAGEM =====")
# for _, row in results_r2_stress.iterrows():
#     print(f"{row['Stress']}: R² = {row['R2'] * 100:.2f}%")

# # ============================================================
# # PONTO 67 - Comparação dos coeficientes da regressão
# # ============================================================

# import numpy as np
# import pandas as pd
# from scipy.stats import linregress

# stress_order = ["Stress baixo", "Stress médio", "Stress alto"]
# regression_results = []

# for stress in stress_order:

#     group = active_pbe_analysis[
#         active_pbe_analysis["stress_voltage"] == stress
#     ].dropna(subset=["PBE", "MAE_active"])

#     x = group["PBE"]
#     y = group["MAE_active"]

#     # Ajustar a regressão linear
#     reg_stress = linregress(x, y)

#     # Previsões e erros
#     y_pred = reg_stress.intercept + reg_stress.slope * x
#     residuals = y - y_pred

#     # R²
#     r2 = reg_stress.rvalue ** 2

#     regression_results.append({
#         "Stress": stress,
#         "N": len(group),
#         "Slope": reg_stress.slope,
#         "Intercept": reg_stress.intercept,
#         "Correlation": reg_stress.rvalue,
#         "R2": r2,
#         "MAE_regression": np.mean(np.abs(residuals)),
#         "RMSE_regression": np.sqrt(np.mean(residuals ** 2))
#     })

# regression_comparison = pd.DataFrame(regression_results)

# print("\n===== PONTO 67 - COMPARAÇÃO DAS REGRESSÕES =====")
# print(regression_comparison.round(8).to_string(index=False))

# # Representação das três retas no mesmo gráfico
# import matplotlib.pyplot as plt

# colors = {
#     "Stress baixo": "green",
#     "Stress médio": "orange",
#     "Stress alto": "red"
# }

# plt.figure(figsize=(10, 6))

# for stress in stress_order:

#     group = active_pbe_analysis[
#         active_pbe_analysis["stress_voltage"] == stress
#     ]

#     row = regression_comparison[
#         regression_comparison["Stress"] == stress
#     ].iloc[0]

#     plt.scatter(
#         group["PBE"],
#         group["MAE_active"],
#         alpha=0.15,
#         s=10,
#         color=colors[stress],
#         label=f"{stress} - dados"
#     )

#     x_line = np.linspace(group["PBE"].min(), group["PBE"].max(), 200)
#     y_line = row["Intercept"] + row["Slope"] * x_line

#     plt.plot(
#         x_line,
#         y_line,
#         color=colors[stress],
#         linewidth=2,
#         label=f"{stress} - regressão"
#     )

# plt.xlabel("PBE")
# plt.ylabel("MAE do active residual (MW)")
# plt.title("Comparação das regressões lineares por nível de stress")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# # ============================================================
# # PONTO 68 - Teste estatístico dos declives
# # ============================================================

# import pandas as pd
# import numpy as np
# import statsmodels.formula.api as smf

# # Preparar os dados
# regression_data = active_pbe_analysis[
#     ["PBE", "MAE_active", "stress_voltage"]
# ].dropna().copy()

# regression_data["stress_voltage"] = (
#     regression_data["stress_voltage"].astype(str)
# )

# # Modelo 1: declive comum para todos os níveis de stress
# model_common = smf.ols(
#     "MAE_active ~ PBE + C(stress_voltage)",
#     data=regression_data
# ).fit()

# # Modelo 2: declives diferentes por nível de stress
# model_interaction = smf.ols(
#     "MAE_active ~ PBE * C(stress_voltage)",
#     data=regression_data
# ).fit()

# # Testar se a interação melhora significativamente o modelo
# comparison = model_interaction.compare_f_test(model_common)

# print("\n===== PONTO 68 - TESTE DOS DECLIVES =====")

# print("\nModelo com declive comum:")
# print(f"R² = {model_common.rsquared:.6f}")

# print("\nModelo com declives específicos por stress:")
# print(f"R² = {model_interaction.rsquared:.6f}")

# print("\nTeste F para a interação:")
# print(f"Estatística F = {comparison[0]:.6f}")
# print(f"p-value = {comparison[1]:.6e}")
# print(f"Graus de liberdade adicionais = {comparison[2]:.0f}")

# alpha = 0.05

# if comparison[1] < alpha:
#     print(
#         "\nConclusão: existem evidências estatísticas de que "
#         "os declives não são todos iguais (p < 0,05)."
#     )
# else:
#     print(
#         "\nConclusão: não existem evidências estatísticas "
#         "suficientes para afirmar que os declives diferem (p >= 0,05)."
#     )

# # ============================================================
# # PONTO 69 - Validação com erros-padrão robustos HC3
# # ============================================================

# import statsmodels.formula.api as smf

# # Preparar os dados
# regression_data = active_pbe_analysis[
#     ["PBE", "MAE_active", "stress_voltage"]
# ].dropna().copy()

# regression_data["stress_voltage"] = (
#     regression_data["stress_voltage"].astype(str)
# )

# # Modelo com declives diferentes por nível de stress
# model_hc3 = smf.ols(
#     "MAE_active ~ PBE * C(stress_voltage)",
#     data=regression_data
# ).fit(cov_type="HC3")

# print("\n===== PONTO 69 - REGRESSÃO COM ERROS-PADRÃO ROBUSTOS =====")
# print(model_hc3.summary())

# # Teste conjunto das interações
# interaction_names = [
#     name for name in model_hc3.params.index
#     if ":" in name and "PBE" in name
# ]

# print("\n===== TESTE CONJUNTO DAS INTERAÇÕES =====")

# hypothesis = " = 0, ".join(interaction_names) + " = 0"

# robust_test = model_hc3.wald_test(
#     hypothesis,
#     scalar=True
# )

# print(f"Estatística Wald: {float(robust_test.statistic):.6f}")
# print(f"p-value: {float(robust_test.pvalue):.6e}")

# if robust_test.pvalue < 0.05:
#     print(
#         "Conclusão: as interações são estatisticamente significativas "
#         "com erros-padrão robustos (p < 0,05)."
#     )
# else:
#     print(
#         "Conclusão: não há evidência estatística suficiente "
#         "de diferenças entre os declives após a correção robusta."
#     )

# # ===== PONTO 70 - INFLUÊNCIA DOS CENÁRIOS EXTREMOS =====

# import statsmodels.formula.api as smf
# import numpy as np

# regression_data = active_pbe_analysis[
#     ["PBE", "MAE_active", "stress_voltage"]
# ].dropna().copy()

# regression_data["stress_voltage"] = (
#     regression_data["stress_voltage"].astype(str)
# )

# # Modelo original, com erros-padrão robustos HC3
# model_original = smf.ols(
#     "MAE_active ~ PBE * C(stress_voltage)",
#     data=regression_data
# ).fit(cov_type="HC3")

# # Identificar os 1% de cenários com maior MAE do resíduo ativo
# limite = regression_data["MAE_active"].quantile(0.99)

# dados_filtrados = regression_data[
#     regression_data["MAE_active"] <= limite
# ].copy()

# # Reajustar o modelo sem esses cenários
# model_filtrado = smf.ols(
#     "MAE_active ~ PBE * C(stress_voltage)",
#     data=dados_filtrados
# ).fit(cov_type="HC3")

# print("===== MODELO ORIGINAL =====")
# print("Número de cenários:", len(regression_data))
# print("R²:", model_original.rsquared)

# print("\n===== MODELO SEM O 1% DE MAIORES MAE =====")
# print("Limite de MAE:", limite)
# print("Número de cenários:", len(dados_filtrados))
# print("R²:", model_filtrado.rsquared)

# print("\n===== COMPARAÇÃO DOS COEFICIENTES =====")
# comparacao = pd.DataFrame({
#     "Original": model_original.params,
#     "Sem 1% extremo": model_filtrado.params
# })

# comparacao["Diferença"] = (
#     comparacao["Sem 1% extremo"] - comparacao["Original"]
# )

# print(comparacao)

# # ===== PONTO 71 - TESTE ROBUSTO DAS INTERAÇÕES SEM EXTREMOS =====

# import numpy as np

# # Ajustar o modelo com interações e erros-padrão robustos
# model_filtrado_hc3 = smf.ols(
#     "MAE_active ~ PBE * C(stress_voltage)",
#     data=dados_filtrados
# ).fit(cov_type="HC3")

# # Identificar os coeficientes das interações
# interaction_idx = [
#     i for i, name in enumerate(model_filtrado_hc3.params.index)
#     if ":" in name and "PBE" in name
# ]

# # Construir a matriz do teste conjunto
# R = np.zeros((len(interaction_idx), len(model_filtrado_hc3.params)))

# for row, idx in enumerate(interaction_idx):
#     R[row, idx] = 1

# # Testar se as duas interações são simultaneamente nulas
# teste = model_filtrado_hc3.wald_test(R, scalar=True)

# print("===== PONTO 71 =====")
# print("Estatística de Wald:", float(teste.statistic))
# print("p-value:", float(teste.pvalue))

# if teste.pvalue < 0.05:
#     print("Conclusão: as interações continuam significativas.")
# else:
#     print("Conclusão: não há evidência suficiente de interações significativas.")

# # ===== PONTO 72 - SIGNIFICÂNCIA INDIVIDUAL DAS INTERAÇÕES =====

# print(model_filtrado_hc3.summary().tables[1])

# # ===== PONTO 73 - TESTE DAS INCLINAÇÕES POR GRUPO =====

# print("Stress alto:")
# print(model_filtrado_hc3.t_test("PBE = 0"))

# print("\nStress baixo:")
# print(model_filtrado_hc3.t_test(
#     "PBE + PBE:C(stress_voltage)[T.Stress baixo] = 0"
# ))

# print("\nStress médio:")
# print(model_filtrado_hc3.t_test(
#     "PBE + PBE:C(stress_voltage)[T.Stress médio] = 0"
# ))

# # ===== PONTO 74 - STRESS MÉDIO VS. STRESS BAIXO =====

# teste_medio_baixo = model_filtrado_hc3.t_test(
#     "PBE:C(stress_voltage)[T.Stress médio] "
#     "- PBE:C(stress_voltage)[T.Stress baixo] = 0"
# )

# print(teste_medio_baixo)

# # ===== PONTO 75 - RESUMO DAS COMPARAÇÕES ENTRE INCLINAÇÕES =====

# comparacoes = pd.DataFrame({
#     "Comparação": [
#         "Stress baixo vs. alto",
#         "Stress médio vs. alto",
#         "Stress médio vs. baixo"
#     ],
#     "Diferença": [
#         0.097892,
#         0.186437,
#         0.088545
#     ],
#     "p-value": [
#         0.095,
#         0.000,
#         0.112
#     ]
# })

# comparacoes["Significativo (5%)"] = (
#     comparacoes["p-value"] < 0.05
# )

# print(comparacoes.to_string(index=False))

# # ===== PONTO 76 - CORREÇÃO DE HOLM =====

# from statsmodels.stats.multitest import multipletests
# import pandas as pd

# comparacoes = pd.DataFrame({
#     "Comparação": [
#         "Stress baixo vs. alto",
#         "Stress médio vs. alto",
#         "Stress médio vs. baixo"
#     ],
#     "p-value": [
#         0.095,  # Stress baixo vs. alto
#         0.000,  # Substituir pelo p-value exato, se disponível
#         0.112   # Stress médio vs. baixo
#     ]
# })

# # Aplicar correção de Holm
# rejeitar, p_holm, _, _ = multipletests(
#     comparacoes["p-value"],
#     alpha=0.05,
#     method="holm"
# )

# comparacoes["p-value Holm"] = p_holm
# comparacoes["Significativo após Holm"] = rejeitar

# print("\n===== PONTO 76 - RESULTADOS =====")
# print(comparacoes.to_string(index=False))

# # ===== PONTO 77 - DISTÂNCIA DE COOK =====

# import statsmodels.formula.api as smf
# import numpy as np
# import pandas as pd

# regression_data = active_pbe_analysis[
#     ["PBE", "MAE_active", "stress_voltage"]
# ].dropna().copy()

# regression_data["stress_voltage"] = (
#     regression_data["stress_voltage"].astype(str)
# )

# # Modelo completo, sem excluir cenários extremos
# model_cook = smf.ols(
#     "MAE_active ~ PBE * C(stress_voltage)",
#     data=regression_data
# ).fit()

# # Medidas de influência
# influence = model_cook.get_influence()
# cooks_d = influence.cooks_distance[0]

# regression_data["cooks_distance"] = cooks_d

# # Critério indicativo: distância de Cook > 4/n
# limite_cook = 4 / len(regression_data)

# influentes = regression_data[
#     regression_data["cooks_distance"] > limite_cook
# ].copy()

# print("===== PONTO 77 - DISTÂNCIA DE COOK =====")
# print("Número de cenários:", len(regression_data))
# print("Limite indicativo (4/n):", limite_cook)
# print("Cenários acima do limite:", len(influentes))
# print(
#     "Percentagem acima do limite:",
#     100 * len(influentes) / len(regression_data),
#     "%"
# )

# print("\n===== 10 MAIORES DISTÂNCIAS DE COOK =====")
# print(
#     regression_data.nlargest(10, "cooks_distance")[
#         ["PBE", "MAE_active", "stress_voltage", "cooks_distance"]
#     ].to_string()
# )

# # ===== PONTO 78 - SENSIBILIDADE À DISTÂNCIA DE COOK =====

# # Modelo original: todos os cenários
# modelo_original_cook = smf.ols(
#     "MAE_active ~ PBE * C(stress_voltage)",
#     data=regression_data
# ).fit(cov_type="HC3")

# # Remover temporariamente os cenários acima do limite
# dados_sem_influentes = regression_data[
#     regression_data["cooks_distance"] <= limite_cook
# ].copy()

# # Modelo de sensibilidade
# modelo_sem_influentes = smf.ols(
#     "MAE_active ~ PBE * C(stress_voltage)",
#     data=dados_sem_influentes
# ).fit(cov_type="HC3")

# # Comparar coeficientes
# comparacao_cook = pd.DataFrame({
#     "Original": modelo_original_cook.params,
#     "Sem influentes": modelo_sem_influentes.params
# })

# comparacao_cook["Variação absoluta"] = (
#     comparacao_cook["Sem influentes"]
#     - comparacao_cook["Original"]
# )

# comparacao_cook["Variação relativa (%)"] = (
#     100
#     * comparacao_cook["Variação absoluta"]
#     / modelo_original_cook.params.replace(0, np.nan)
# )

# print("===== PONTO 78 =====")
# print("Cenários originais:", len(regression_data))
# print("Cenários após filtragem:", len(dados_sem_influentes))
# print("R² original:", modelo_original_cook.rsquared)
# print("R² sem influentes:", modelo_sem_influentes.rsquared)

# print("\n===== COMPARAÇÃO DOS COEFICIENTES =====")
# print(comparacao_cook.to_string())

# # ===== PONTO 79 - TESTE ROBUSTO APÓS DISTÂNCIA DE COOK =====

# modelo_sem_influentes_hc3 = smf.ols(
#     "MAE_active ~ PBE * C(stress_voltage)",
#     data=dados_sem_influentes
# ).fit(cov_type="HC3")

# # Teste conjunto das duas interações
# interaction_idx = [
#     i for i, name in enumerate(modelo_sem_influentes_hc3.params.index)
#     if ":" in name and "PBE" in name
# ]

# R = np.zeros((
#     len(interaction_idx),
#     len(modelo_sem_influentes_hc3.params)
# ))

# for row, idx in enumerate(interaction_idx):
#     R[row, idx] = 1

# teste_cook = modelo_sem_influentes_hc3.wald_test(R, scalar=True)

# print("===== PONTO 79 =====")
# print("Estatística de Wald:", float(teste_cook.statistic))
# print("p-value:", float(teste_cook.pvalue))
# print("R²:", modelo_sem_influentes_hc3.rsquared)

# if teste_cook.pvalue < 0.05:
#     print("Conclusão: as interações continuam significativas.")
# else:
#     print("Conclusão: as interações deixam de ser significativas.")

# # ===== PONTO 80 - TESTE DE BREUSCH-PAGAN =====

# from statsmodels.stats.diagnostic import het_breuschpagan

# # Modelo OLS original, sem remover cenários
# modelo_bp = smf.ols(
#     "MAE_active ~ PBE * C(stress_voltage)",
#     data=regression_data
# ).fit()

# # Teste de Breusch-Pagan
# bp = het_breuschpagan(
#     modelo_bp.resid,
#     modelo_bp.model.exog
# )

# print("===== PONTO 80 =====")
# print("Estatística LM:", bp[0])
# print("p-value LM:", bp[1])
# print("Estatística F:", bp[2])
# print("p-value F:", bp[3])

# # ===== PONTO 81 =====
# import matplotlib.pyplot as plt

# modelo_bp = smf.ols(
#     "MAE_active ~ PBE * C(stress_voltage)",
#     data=regression_data
# ).fit()

# plt.figure(figsize=(10, 6))

# for stress in regression_data["stress_voltage"].unique():
#     mask = regression_data["stress_voltage"] == stress

#     plt.scatter(
#         modelo_bp.fittedvalues[mask],
#         modelo_bp.resid[mask],
#         label=stress,
#         alpha=0.45,
#         s=18
#     )

# plt.axhline(0, linestyle="--", color="black")

# plt.xlabel("Valores previstos de MAE_active")
# plt.ylabel("Resíduos")
# plt.title("Resíduos vs. valores previstos — modelo PBE e stress")
# plt.legend(title="Stress")
# plt.grid(alpha=0.25)
# plt.tight_layout()
# plt.show()

# # ===== PONTO 82 =====


# print("===== PONTO 82 — Comparação dos modelos =====")

# print("\nModelo original:")
# print(model_cook.params)

# print("\nModelo sem observações influentes:")
# print(modelo_sem_influentes.params)

# print("\nDiferença entre coeficientes:")
# print(modelo_sem_influentes.params - model_cook.params)


# # ===== PONTO 83 — Inclinações por nível de stress =====

# print("===== PONTO 83 — Inclinações por nível de stress =====")

# modelo_hc3_sem_influentes = modelo_sem_influentes.get_robustcov_results(
#     cov_type="HC3"
# )

# nomes = modelo_sem_influentes.model.exog_names
# print("Coeficientes:", nomes)

# print("\nResumo com erros-padrão robustos HC3:")
# print(modelo_hc3_sem_influentes.summary())


# # ===== PONTO 84 — Comparação stress médio vs. baixo =====

# print("===== PONTO 84 — Comparação stress médio vs. baixo =====")

# teste_medio_baixo = modelo_hc3_sem_influentes.t_test(
#     "PBE:C(stress_voltage)[T.Stress médio] = "
#     "PBE:C(stress_voltage)[T.Stress baixo]"
# )

# print(teste_medio_baixo)


# # ===== PONTO 85 — Teste conjunto das interações =====

# print("===== PONTO 85 — Teste conjunto das interações =====")

# teste_interacoes = modelo_hc3_sem_influentes.wald_test(
#     "PBE:C(stress_voltage)[T.Stress baixo] = 0, "
#     "PBE:C(stress_voltage)[T.Stress médio] = 0",
#     scalar=True
# )

# print(teste_interacoes)


# # ===== PONTO 86 — Comparação do ajustamento =====
# print("===== PONTO 86 — Comparação do ajustamento =====")

# print(f"R² modelo original: {model_cook.rsquared:.6f}")
# print(f"R² sem observações influentes: {modelo_sem_influentes.rsquared:.6f}")

# diferenca_r2 = model_cook.rsquared - modelo_sem_influentes.rsquared

# print(f"Diferença absoluta de R²: {diferenca_r2:.6f}")
# print(f"Variação relativa de R²: {100 * diferenca_r2 / model_cook.rsquared:.2f}%")


# # ===== PONTO 87.1 — Divisão treino/teste =====

# from sklearn.model_selection import train_test_split

# print("===== PONTO 87.1 — Divisão treino/teste =====")

# dados_ml = regression_data[
#     ["MAE_active", "PBE", "stress_voltage"]
# ].dropna().copy()

# treino, teste = train_test_split(
#     dados_ml,
#     test_size=0.20,
#     random_state=42
# )

# print(f"Total de cenários: {len(dados_ml)}")
# print(f"Cenários de treino: {len(treino)}")
# print(f"Cenários de teste: {len(teste)}")

# print("\nDistribuição do stress no treino:")
# print(treino["stress_voltage"].value_counts())

# print("\nDistribuição do stress no teste:")
# print(teste["stress_voltage"].value_counts())


# # ===== PONTO 87.2 — Treino e teste =====

# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np
# import statsmodels.formula.api as smf

# print("===== PONTO 87.2 — Treino e teste =====")

# modelo_treino = smf.ols(
#     "MAE_active ~ PBE * C(stress_voltage)",
#     data=treino
# ).fit()

# # Previsões nos dois conjuntos
# pred_treino = modelo_treino.predict(treino)
# pred_teste = modelo_treino.predict(teste)

# # Métricas no treino
# mae_treino = mean_absolute_error(
#     treino["MAE_active"], pred_treino
# )
# rmse_treino = np.sqrt(
#     mean_squared_error(treino["MAE_active"], pred_treino)
# )
# r2_treino = r2_score(
#     treino["MAE_active"], pred_treino
# )

# # Métricas no teste
# mae_teste = mean_absolute_error(
#     teste["MAE_active"], pred_teste
# )
# rmse_teste = np.sqrt(
#     mean_squared_error(teste["MAE_active"], pred_teste)
# )
# r2_teste = r2_score(
#     teste["MAE_active"], pred_teste
# )

# print("\n--- Treino ---")
# print(f"MAE:  {mae_treino:.8f}")
# print(f"RMSE: {rmse_treino:.8f}")
# print(f"R²:   {r2_treino:.6f}")

# print("\n--- Teste ---")
# print(f"MAE:  {mae_teste:.8f}")
# print(f"RMSE: {rmse_teste:.8f}")
# print(f"R²:   {r2_teste:.6f}")


# # ===== PONTO 88 — Desempenho por nível de stress =====

# print("===== PONTO 88 — Desempenho por nível de stress =====")

# avaliacao_teste = teste.copy()
# avaliacao_teste["MAE_pred"] = pred_teste

# avaliacao_teste["erro_absoluto"] = np.abs(
#     avaliacao_teste["MAE_active"] -
#     avaliacao_teste["MAE_pred"]
# )

# avaliacao_teste["erro_quadratico"] = (
#     avaliacao_teste["MAE_active"] -
#     avaliacao_teste["MAE_pred"]
# ) ** 2

# resultado_stress = avaliacao_teste.groupby(
#     "stress_voltage", observed=True
# ).apply(
#     lambda grupo: pd.Series({
#         "N": len(grupo),
#         "MAE_teste": grupo["erro_absoluto"].mean(),
#         "RMSE_teste": np.sqrt(
#             grupo["erro_quadratico"].mean()
#         ),
#         "R2_teste": r2_score(
#             grupo["MAE_active"],
#             grupo["MAE_pred"]
#         )
#     }),
#     include_groups=False
# )

# print(resultado_stress)


# # ===== PONTO 89 — Viés das previsões por stress =====

# print("===== PONTO 89 — Viés das previsões por stress =====")

# avaliacao_teste["erro_assinado"] = (
#     avaliacao_teste["MAE_pred"] -
#     avaliacao_teste["MAE_active"]
# )

# resultado_vies = avaliacao_teste.groupby(
#     "stress_voltage", observed=True
# ).agg(
#     N=("erro_assinado", "size"),
#     Erro_medio=("erro_assinado", "mean"),
#     Erro_mediano=("erro_assinado", "median"),
#     Erro_abs_medio=("erro_absoluto", "mean")
# )

# print(resultado_vies)


# # ===== PONTO 90 — Dez maiores erros no teste =====

# print("===== PONTO 90 — Dez maiores erros no teste =====")

# colunas_resultado = [
#     "MAE_active",
#     "PBE",
#     "stress_voltage",
#     "MAE_pred",
#     "erro_assinado",
#     "erro_absoluto"
# ]

# maiores_erros = (
#     avaliacao_teste[colunas_resultado]
#     .sort_values("erro_absoluto", ascending=False)
#     .head(10)
# )

# print(maiores_erros.to_string())



# # ===== PONTO 91 — Correlação entre PBE e erro de previsão =====

# print("===== PONTO 91 — Correlação entre PBE e erro de previsão =====")

# print("\nCorrelação global:")
# print(
#     avaliacao_teste[["PBE", "erro_absoluto"]]
#     .corr(method="pearson")
#     .iloc[0, 1]
# )

# print("\nCorrelação por nível de stress:")

# for stress, grupo in avaliacao_teste.groupby(
#     "stress_voltage", observed=True
# ):
#     correlacao = grupo["PBE"].corr(
#         grupo["erro_absoluto"],
#         method="pearson"
#     )

#     print(f"{stress}: r = {correlacao:.4f}")


# # ===== PONTO 92 — PBE vs. erro absoluto =====

# import matplotlib.pyplot as plt

# print("===== PONTO 92 — PBE vs. erro absoluto =====")

# plt.figure(figsize=(10, 6))

# for stress, grupo in avaliacao_teste.groupby(
#     "stress_voltage", observed=True
# ):
#     plt.scatter(
#         grupo["PBE"],
#         grupo["erro_absoluto"],
#         label=stress,
#         alpha=0.55,
#         s=20
#     )

# plt.xlabel("PBE")
# plt.ylabel("Erro absoluto de previsão de MAE_active")
# plt.title("PBE vs. erro de previsão por nível de stress")
# plt.legend(title="Nível de stress")
# plt.grid(alpha=0.25)
# plt.tight_layout()
# plt.show()


# # ===== PONTO 93 — Comparação dos erros por stress =====

# from scipy.stats import mannwhitneyu

# print("===== PONTO 93 — Comparação dos erros por stress =====")

# alto = avaliacao_teste.loc[
#     avaliacao_teste["stress_voltage"] == "Stress alto",
#     "erro_absoluto"
# ].dropna()

# medio = avaliacao_teste.loc[
#     avaliacao_teste["stress_voltage"] == "Stress médio",
#     "erro_absoluto"
# ].dropna()

# baixo = avaliacao_teste.loc[
#     avaliacao_teste["stress_voltage"] == "Stress baixo",
#     "erro_absoluto"
# ].dropna()

# comparacoes = [
#     ("Alto vs. médio", alto, medio),
#     ("Alto vs. baixo", alto, baixo),
#     ("Médio vs. baixo", medio, baixo)
# ]

# for nome, grupo1, grupo2 in comparacoes:
#     resultado = mannwhitneyu(
#         grupo1,
#         grupo2,
#         alternative="two-sided"
#     )

#     print(f"\n{nome}")
#     print(f"Mediana grupo 1: {grupo1.median():.8f}")
#     print(f"Mediana grupo 2: {grupo2.median():.8f}")
#     print(f"Estatística U: {resultado.statistic:.2f}")
#     print(f"p-value: {resultado.pvalue:.6e}")


# # ===== PONTO 94 — Dimensão das diferenças =====

# print("===== PONTO 94 — Dimensão das diferenças =====")

# medianas = {
#     stress: grupo["erro_absoluto"].median()
#     for stress, grupo in avaliacao_teste.groupby(
#         "stress_voltage", observed=True
#     )
# }

# comparacoes = [
#     ("Alto", "Médio"),
#     ("Alto", "Baixo"),
#     ("Médio", "Baixo")
# ]

# for grupo1, grupo2 in comparacoes:
#     med1 = medianas[f"Stress {grupo1.lower()}"]
#     med2 = medianas[f"Stress {grupo2.lower()}"]

#     razao = med1 / med2
#     diferenca = med1 - med2
#     variacao_pct = (razao - 1) * 100

#     print(f"\nStress {grupo1} vs. Stress {grupo2}")
#     print(f"Razão das medianas: {razao:.3f}x")
#     print(f"Diferença absoluta: {diferenca:.8f}")
#     print(f"Variação relativa: {variacao_pct:.2f}%")


# # ==== PONTO 95 — Resumo da avaliação =====

# print("===== PONTO 95 — Resumo da avaliação =====")

# print("\nMétricas globais:")
# resumo_global = pd.DataFrame({
#     "Conjunto": ["Treino", "Teste"],
#     "MAE": [mae_treino, mae_teste],
#     "RMSE": [rmse_treino, rmse_teste],
#     "R2": [r2_treino, r2_teste]
# })

# print(resumo_global.to_string(index=False))

# print("\nMétricas por nível de stress:")
# print(resultado_stress.to_string())

# print("\nMedianas dos erros absolutos:")
# print(
#     avaliacao_teste.groupby(
#         "stress_voltage", observed=True
#     )["erro_absoluto"].median()
# )


# # ==== PONTO 96.1 — Nova divisão treino/teste =====

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np

# print("===== PONTO 96.1 — Nova divisão treino/teste =====")

# treino_123, teste_123 = train_test_split(
#     dados_ml,
#     test_size=0.20,
#     random_state=123
# )

# modelo_123 = smf.ols(
#     "MAE_active ~ PBE * C(stress_voltage)",
#     data=treino_123
# ).fit()

# pred_teste_123 = modelo_123.predict(teste_123)

# mae_teste_123 = mean_absolute_error(
#     teste_123["MAE_active"],
#     pred_teste_123
# )

# rmse_teste_123 = np.sqrt(
#     mean_squared_error(
#         teste_123["MAE_active"],
#         pred_teste_123
#     )
# )

# r2_teste_123 = r2_score(
#     teste_123["MAE_active"],
#     pred_teste_123
# )

# print(f"N.º de cenários de teste: {len(teste_123)}")
# print(f"MAE:  {mae_teste_123:.8f}")
# print(f"RMSE: {rmse_teste_123:.8f}")
# print(f"R²:   {r2_teste_123:.6f}")



# # ============================================================
# # PONTO 96.2 — Avaliação com 5 divisões aleatórias
# # ============================================================

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import statsmodels.formula.api as smf
# import numpy as np
# import pandas as pd

# dados_ml = regression_data[
#     ["MAE_active", "PBE", "stress_voltage"]
# ].dropna().copy()

# resultados = []

# for seed in [42, 123, 2026, 7, 99]:

#     treino, teste = train_test_split(
#         dados_ml,
#         test_size=0.20,
#         random_state=seed
#     )

#     modelo = smf.ols(
#         "MAE_active ~ PBE * C(stress_voltage)",
#         data=treino
#     ).fit()

#     previsoes = modelo.predict(teste)
#     valores_reais = teste["MAE_active"]

#     mae = mean_absolute_error(valores_reais, previsoes)
#     rmse = np.sqrt(mean_squared_error(valores_reais, previsoes))
#     r2 = r2_score(valores_reais, previsoes)

#     resultados.append({
#         "seed": seed,
#         "N_treino": len(treino),
#         "N_teste": len(teste),
#         "MAE": mae,
#         "RMSE": rmse,
#         "R2": r2
#     })

# resultados_df = pd.DataFrame(resultados)

# print("\nPONTO 96.2 — Resultados das 5 divisões:")
# print(resultados_df.to_string(index=False, float_format=lambda x: f"{x:.8f}"))

# print("\nResumo estatístico:")
# print(
#     resultados_df[["MAE", "RMSE", "R2"]]
#     .agg(["mean", "std"])
#     .to_string(float_format=lambda x: f"{x:.8f}")
# )



# # ============================================================
# # PONTO 96.3 — Desempenho por nível de stress
# # nas cinco divisões treino/teste
# # ============================================================

# resultados_stress = []

# for seed in [42, 123, 2026, 7, 99]:

#     treino, teste = train_test_split(
#         dados_ml,
#         test_size=0.20,
#         random_state=seed
#     )

#     modelo = smf.ols(
#         "MAE_active ~ PBE * C(stress_voltage)",
#         data=treino
#     ).fit()

#     teste = teste.copy()
#     teste["MAE_pred"] = modelo.predict(teste)

#     for stress, grupo in teste.groupby("stress_voltage"):

#         y_real = grupo["MAE_active"]
#         y_pred = grupo["MAE_pred"]

#         resultados_stress.append({
#             "seed": seed,
#             "stress": stress,
#             "N_teste": len(grupo),
#             "MAE": mean_absolute_error(y_real, y_pred),
#             "RMSE": np.sqrt(mean_squared_error(y_real, y_pred)),
#             "R2": (
#                 r2_score(y_real, y_pred)
#                 if len(grupo) > 1 and y_real.nunique() > 1
#                 else np.nan
#             )
#         })

# resultados_stress_df = pd.DataFrame(resultados_stress)

# print("\nPONTO 96.3 — Resultados por stress e divisão:")
# print(
#     resultados_stress_df
#     .sort_values(["stress", "seed"])
#     .to_string(index=False, float_format=lambda x: f"{x:.8f}")
# )

# print("\nResumo por nível de stress:")
# print(
#     resultados_stress_df
#     .groupby("stress")[["MAE", "RMSE", "R2"]]
#     .agg(["mean", "std"])
#     .to_string(float_format=lambda x: f"{x:.8f}")
# )




# # ============================================================
# # PONTO 96.4 — Análise de erros extremos por nível de stress
# # ============================================================

# resultados_extremos = []

# for seed in [42, 123, 2026, 7, 99]:

#     treino, teste = train_test_split(
#         dados_ml,
#         test_size=0.20,
#         random_state=seed
#     )

#     modelo = smf.ols(
#         "MAE_active ~ PBE * C(stress_voltage)",
#         data=treino
#     ).fit()

#     teste = teste.copy()
#     teste["MAE_pred"] = modelo.predict(teste)

#     # Erro absoluto entre previsão e valor real
#     teste["erro_abs_pred"] = (
#         teste["MAE_pred"] - teste["MAE_active"]
#     ).abs()

#     for stress, grupo in teste.groupby("stress_voltage"):

#         erro = grupo["erro_abs_pred"]

#         resultados_extremos.append({
#             "seed": seed,
#             "stress": stress,
#             "N_teste": len(grupo),
#             "MAE_erro": erro.mean(),
#             "Mediana": erro.median(),
#             "P95": erro.quantile(0.95),
#             "Maximo": erro.max()
#         })

# resultados_extremos_df = pd.DataFrame(resultados_extremos)

# print("\nPONTO 96.4 — Erros extremos por divisão:")
# print(
#     resultados_extremos_df
#     .sort_values(["stress", "seed"])
#     .to_string(index=False, float_format=lambda x: f"{x:.8f}")
# )

# print("\nResumo por nível de stress:")
# print(
#     resultados_extremos_df
#     .groupby("stress")[["MAE_erro", "Mediana", "P95", "Maximo"]]
#     .agg(["mean", "std"])
#     .to_string(float_format=lambda x: f"{x:.8f}")
# )


# # ============================================================
# # PONTO 96.5 — Identificar cenários com erros extremos
# # ============================================================

# seed = 42

# treino, teste = train_test_split(
#     dados_ml,
#     test_size=0.20,
#     random_state=seed
# )

# modelo = smf.ols(
#     "MAE_active ~ PBE * C(stress_voltage)",
#     data=treino
# ).fit()

# teste = teste.copy()
# teste["MAE_pred"] = modelo.predict(teste)

# # Erro assinado e erro absoluto da previsão
# teste["erro_pred"] = teste["MAE_pred"] - teste["MAE_active"]
# teste["erro_abs_pred"] = teste["erro_pred"].abs()

# # Selecionar os 20 maiores erros absolutos
# top20_erros = teste.nlargest(20, "erro_abs_pred")

# colunas = [
#     "PBE",
#     "stress_voltage",
#     "MAE_active",
#     "MAE_pred",
#     "erro_pred",
#     "erro_abs_pred"
# ]

# print("\nPONTO 96.5 — Top 20 erros absolutos (seed=42):")
# print(
#     top20_erros[colunas].to_string(
#         float_format=lambda x: f"{x:.8f}"
#     )
# )

# print("\nDistribuição dos 20 casos por stress:")
# print(top20_erros["stress_voltage"].value_counts())

# print("\nResumo do PBE nos 20 casos:")
# print(top20_erros["PBE"].describe())



# # ============================================================
# # PONTO 96.6 — Relação entre PBE e MAE_active
# # com curvas polinomiais de grau 2
# # ============================================================

# import matplotlib.pyplot as plt
# import numpy as np

# fig, ax = plt.subplots(figsize=(10, 6))

# cores = {
#     "Stress alto": "red",
#     "Stress médio": "orange",
#     "Stress baixo": "green"
# }

# for stress, grupo in regression_data.groupby("stress_voltage"):

#     x = grupo["PBE"].to_numpy()
#     y = grupo["MAE_active"].to_numpy()

#     # Pontos observados
#     ax.scatter(
#         x, y,
#         alpha=0.20,
#         s=12,
#         color=cores.get(stress, "blue"),
#         label=stress
#     )

#     # Curva polinomial de grau 2
#     if len(np.unique(x)) >= 3:
#         coef = np.polyfit(x, y, deg=2)
#         polinomio = np.poly1d(coef)

#         x_curva = np.linspace(x.min(), x.max(), 300)
#         ordem = np.argsort(x_curva)

#         ax.plot(
#             x_curva[ordem],
#             polinomio(x_curva[ordem]),
#             color=cores.get(stress, "blue"),
#             linewidth=2
#         )

# ax.set_title("PBE vs MAE_active por nível de stress")
# ax.set_xlabel("PBE")
# ax.set_ylabel("MAE_active")
# ax.grid(True, alpha=0.3)
# ax.legend()
# plt.tight_layout()
# plt.show()



# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import statsmodels.formula.api as smf
# import numpy as np
# import pandas as pd

# # Preparar os dados
# dados_ml = regression_data[
#     ["MAE_active", "PBE", "stress_voltage"]
# ].dropna().copy()

# # Usar a mesma divisão treino/teste
# treino, teste = train_test_split(
#     dados_ml,
#     test_size=0.20,
#     random_state=42
# )

# # Modelo linear
# modelo_linear = smf.ols(
#     "MAE_active ~ PBE * C(stress_voltage)",
#     data=treino
# ).fit()

# # Modelo quadrático: acrescenta PBE² e a sua interação com o stress
# modelo_quadratico = smf.ols(
#     "MAE_active ~ (PBE + I(PBE**2)) * C(stress_voltage)",
#     data=treino
# ).fit()

# # Comparar previsões no conjunto de teste
# resultados = []

# for nome, modelo in [
#     ("Linear", modelo_linear),
#     ("Quadrático", modelo_quadratico)
# ]:
#     previsoes = modelo.predict(teste)

#     resultados.append({
#         "Modelo": nome,
#         "MAE": mean_absolute_error(
#             teste["MAE_active"], previsoes
#         ),
#         "RMSE": np.sqrt(mean_squared_error(
#             teste["MAE_active"], previsoes
#         )),
#         "R2": r2_score(
#             teste["MAE_active"], previsoes
#         )
#     })

# print("\nComparação dos modelos no conjunto de teste:")
# print(pd.DataFrame(resultados).to_string(index=False))



# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import statsmodels.formula.api as smf
# import numpy as np
# import pandas as pd

# dados_ml = regression_data[
#     ["MAE_active", "PBE", "stress_voltage"]
# ].dropna().copy()

# resultados = []

# for seed in [42, 123, 2026, 7, 99]:
#     treino, teste = train_test_split(
#         dados_ml,
#         test_size=0.20,
#         random_state=seed
#     )

#     modelos = {
#         "Linear": "MAE_active ~ PBE * C(stress_voltage)",
#         "Quadrático": (
#             "MAE_active ~ (PBE + I(PBE**2)) "
#             "* C(stress_voltage)"
#         )
#     }

#     for nome, formula in modelos.items():
#         modelo = smf.ols(formula, data=treino).fit()
#         previsoes = modelo.predict(teste)

#         resultados.append({
#             "Seed": seed,
#             "Modelo": nome,
#             "MAE": mean_absolute_error(
#                 teste["MAE_active"], previsoes
#             ),
#             "RMSE": np.sqrt(mean_squared_error(
#                 teste["MAE_active"], previsoes
#             )),
#             "R2": r2_score(
#                 teste["MAE_active"], previsoes
#             )
#         })

# resultados_df = pd.DataFrame(resultados)

# print("\nResultados por seed:")
# print(resultados_df.round(8).to_string(index=False))

# print("\nMédia e desvio-padrão:")
# print(
#     resultados_df.groupby("Modelo")[["MAE", "RMSE", "R2"]]
#     .agg(["mean", "std"])
#     .round(8)
# )




# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import statsmodels.formula.api as smf
# import numpy as np
# import pandas as pd

# dados_ml = regression_data[
#     ["MAE_active", "PBE", "stress_voltage"]
# ].dropna().copy()

# resultados_stress = []

# for seed in [42, 123, 2026, 7, 99]:
#     treino, teste = train_test_split(
#         dados_ml,
#         test_size=0.20,
#         random_state=seed
#     )

#     for nome, formula in [
#         ("Linear", "MAE_active ~ PBE * C(stress_voltage)"),
#         ("Quadrático",
#          "MAE_active ~ (PBE + I(PBE**2)) * C(stress_voltage)")
#     ]:
#         modelo = smf.ols(formula, data=treino).fit()
#         previsoes = modelo.predict(teste)

#         for stress, grupo in teste.groupby("stress_voltage"):
#             pred = previsoes.loc[grupo.index]
#             real = grupo["MAE_active"]

#             resultados_stress.append({
#                 "Seed": seed,
#                 "Modelo": nome,
#                 "Stress": stress,
#                 "N": len(grupo),
#                 "MAE": mean_absolute_error(real, pred),
#                 "RMSE": np.sqrt(mean_squared_error(real, pred))
#             })

# stress_df = pd.DataFrame(resultados_stress)

# print("\nResultados por stress e seed:")
# print(stress_df.round(8).to_string(index=False))

# print("\nMédia das métricas por modelo e stress:")
# print(
#     stress_df.groupby(["Modelo", "Stress"])[["MAE", "RMSE"]]
#     .mean()
#     .round(8)
# )



# # Comparação cenário a cenário: seed 42

# treino, teste = train_test_split(
#     dados_ml,
#     test_size=0.20,
#     random_state=42
# )

# modelo_linear = smf.ols(
#     "MAE_active ~ PBE * C(stress_voltage)",
#     data=treino
# ).fit()

# modelo_quadratico = smf.ols(
#     "MAE_active ~ (PBE + I(PBE**2)) * C(stress_voltage)",
#     data=treino
# ).fit()

# comparacao = teste[
#     ["MAE_active", "PBE", "stress_voltage"]
# ].copy()

# comparacao["pred_linear"] = modelo_linear.predict(teste)
# comparacao["pred_quadratico"] = modelo_quadratico.predict(teste)

# comparacao["erro_linear"] = (
#     comparacao["pred_linear"] - comparacao["MAE_active"]
# ).abs()

# comparacao["erro_quadratico"] = (
#     comparacao["pred_quadratico"] - comparacao["MAE_active"]
# ).abs()

# comparacao["melhoria_abs"] = (
#     comparacao["erro_linear"] - comparacao["erro_quadratico"]
# )

# print("Cenários em que o quadrático melhora:",
#       (comparacao["melhoria_abs"] > 0).sum())

# print("Cenários em que o linear melhora:",
#       (comparacao["melhoria_abs"] < 0).sum())

# print("Diferença média do erro absoluto (linear - quadrático):",
#       comparacao["melhoria_abs"].mean())

# print("\n10 maiores melhorias absolutas do quadrático:")
# print(
#     comparacao.sort_values("melhoria_abs", ascending=False)
#     [["MAE_active", "PBE", "stress_voltage",
#       "erro_linear", "erro_quadratico", "melhoria_abs"]]
#     .head(10)
#     .to_string()
# )



# # Comparação da melhoria por nível de stress

# resumo_stress = (
#     comparacao.groupby("stress_voltage")
#     .agg(
#         N=("melhoria_abs", "size"),
#         Quadratico_melhora=(
#             "melhoria_abs",
#             lambda x: (x > 0).sum()
#         ),
#         Linear_melhora=(
#             "melhoria_abs",
#             lambda x: (x < 0).sum()
#         ),
#         Diferenca_media=("melhoria_abs", "mean"),
#         Diferenca_mediana=("melhoria_abs", "median")
#     )
# )

# resumo_stress["Pct_quadratico_melhora"] = (
#     100 * resumo_stress["Quadratico_melhora"]
#     / resumo_stress["N"]
# )

# print("\nComparação por nível de stress:")
# print(resumo_stress.round(8).to_string())



# from scipy.stats import wilcoxon

# print("Teste de Wilcoxon por nível de stress\n")

# for stress, grupo in comparacao.groupby("stress_voltage"):
#     diferencas = (
#         grupo["erro_linear"] - grupo["erro_quadratico"]
#     )

#     estatistica, p_valor = wilcoxon(
#         grupo["erro_linear"],
#         grupo["erro_quadratico"],
#         alternative="two-sided"
#     )

#     print(f"{stress}:")
#     print(f"  N = {len(grupo)}")
#     print(f"  Diferença mediana = {diferencas.median():.8f}")
#     print(f"  Estatística = {estatistica:.3f}")
#     print(f"  p-valor = {p_valor:.6g}\n")


# ## Final


# from pathlib import Path
# import pandas as pd

# ficheiro_14 = Path(
#     "outputs/case14_seed42/case14_ieee_predictions.parquet"
# )

# ficheiro_30 = Path(
#     "outputs/case30_seed42/case30_ieee_predictions.parquet"
# )

# df14 = pd.read_parquet(ficheiro_14)
# df30 = pd.read_parquet(ficheiro_30)

# print("===== IEEE 14 =====")
# print("Dimensões:", df14.shape)
# print("Colunas:", df14.columns.tolist())

# print("\n===== IEEE 30 =====")
# print("Dimensões:", df30.shape)
# print("Colunas:", df30.columns.tolist())

# print("\n===== Comparação das colunas =====")
# print("Colunas comuns:", sorted(set(df14.columns) & set(df30.columns)))
# print("Só no IEEE 14:", sorted(set(df14.columns) - set(df30.columns)))
# print("Só no IEEE 30:", sorted(set(df30.columns) - set(df14.columns)))



# import numpy as np
# import pandas as pd

# # Calcular erros absolutos
# for df in [df14, df30]:
#     df["error_Vm"] = (df["Vm_pred"] - df["Vm_target"]).abs()
#     df["error_Va"] = (df["Va_pred"] - df["Va_target"]).abs()
#     df["error_Pg"] = (df["Pg_pred"] - df["Pg_target"]).abs()
#     df["error_Qg"] = (df["Qg_pred"] - df["Qg_target"]).abs()

# def calcular_metricas(df, rede):
#     resultados = []

#     for variavel in ["Vm", "Va", "Pg", "Qg"]:
#         erros = df[f"error_{variavel}"].dropna()

#         resultados.append({
#             "Rede": rede,
#             "Variavel": variavel,
#             "MAE": erros.mean(),
#             "RMSE": np.sqrt((erros ** 2).mean()),
#             "P95": erros.quantile(0.95),
#             "P99": erros.quantile(0.99),
#             "MAX": erros.max()
#         })

#     return resultados

# metricas = pd.DataFrame(
#     calcular_metricas(df14, "IEEE 14") +
#     calcular_metricas(df30, "IEEE 30")
# )

# print("\n===== COMPARAÇÃO GLOBAL =====")
# print(metricas.round(8).to_string(index=False))



# for nome, df in [("IEEE 14", df14), ("IEEE 30", df30)]:
#     # Barramentos com geração: PV ou referência
#     geradores = df[(df["PV"] == 1) | (df["REF"] == 1)]

#     print(f"\n===== {nome} — BARRAMENTOS COM GERAÇÃO =====")
#     print("Número de linhas:", len(geradores))
#     print("Barramentos:", sorted(geradores["bus"].unique()))

#     for variavel in ["Pg", "Qg"]:
#         erros = geradores[f"error_{variavel}"].dropna()

#         print(
#             f"{variavel}: "
#             f"MAE={erros.mean():.6f}, "
#             f"RMSE={np.sqrt((erros**2).mean()):.6f}, "
#             f"P99={erros.quantile(0.99):.6f}, "
#             f"MAX={erros.max():.6f}"
#         )



# # Identificar os maiores erros de Pg nos barramentos com geração

# for nome, df in [("IEEE 14", df14), ("IEEE 30", df30)]:
#     geradores = df[(df["PV"] == 1) | (df["REF"] == 1)]

#     top_erros = (
#         geradores.sort_values("error_Pg", ascending=False)
#         [["scenario", "bus", "Pg_target", "Pg_pred", "error_Pg"]]
#         .head(10)
#     )

#     print(f"\n===== {nome}: TOP 10 ERROS DE Pg =====")
#     print(top_erros.to_string(index=False))

#     print("\nNúmero de erros acima de 1 MW:",
#           (geradores["error_Pg"] > 1).sum())


# # Analisar a compensação de potência ativa no cenário com maior erro

# for nome, df, scenario_id in [
#     ("IEEE 14", df14, 89847),
#     ("IEEE 30", df30, 239279)
# ]:
#     geradores = df[
#         (df["scenario"] == scenario_id) &
#         ((df["PV"] == 1) | (df["REF"] == 1))
#     ].copy()

#     print(f"\n===== {nome} | Cenário {scenario_id} =====")
#     print(
#         geradores[
#             ["bus", "Pg_target", "Pg_pred", "error_Pg"]
#         ].to_string(index=False)
#     )

#     print("\nSoma de Pg de referência:",
#           geradores["Pg_target"].sum(), "MW")
#     print("Soma de Pg prevista:",
#           geradores["Pg_pred"].sum(), "MW")
#     print("Erro absoluto na soma:",
#           abs(geradores["Pg_pred"].sum() -
#               geradores["Pg_target"].sum()), "MW")



# # Avaliar a compensação dos erros de Pg em todos os cenários

# for nome, df in [("IEEE 14", df14), ("IEEE 30", df30)]:
#     geradores = df[(df["PV"] == 1) | (df["REF"] == 1)]

#     analise = geradores.groupby("scenario").agg(
#         Pg_target_total=("Pg_target", "sum"),
#         Pg_pred_total=("Pg_pred", "sum"),
#         soma_erros_abs=("error_Pg", "sum")
#     )

#     analise["erro_total_abs"] = (
#         analise["Pg_pred_total"] - analise["Pg_target_total"]
#     ).abs()

#     analise["compensacao"] = (
#         1 - analise["erro_total_abs"] /
#         analise["soma_erros_abs"].replace(0, float("nan"))
#     )

#     print(f"\n===== {nome} =====")
#     print("Erro absoluto médio na soma (MW):",
#           analise["erro_total_abs"].mean())
#     print("Soma média dos erros absolutos individuais (MW):",
#           analise["soma_erros_abs"].mean())
#     print("Compensação média (%):",
#           analise["compensacao"].mean() * 100)



# for nome, df in [("IEEE 14", df14), ("IEEE 30", df30)]:
#     geradores = df[(df["PV"] == 1) | (df["REF"] == 1)]

#     por_cenario = geradores.groupby("scenario").agg(
#         Pg_target=("Pg_target", "sum"),
#         Pg_pred=("Pg_pred", "sum")
#     )

#     erro_total = (por_cenario["Pg_pred"] - por_cenario["Pg_target"]).abs()

#     print(f"\n===== {nome} =====")
#     print("Mediana:", erro_total.median(), "MW")
#     print("Percentil 95:", erro_total.quantile(0.95), "MW")
#     print("Percentil 99:", erro_total.quantile(0.99), "MW")
#     print("Máximo:", erro_total.max(), "MW")



# for nome, df in [("IEEE 14", df14), ("IEEE 30", df30)]:
#     geradores = df[(df["PV"] == 1) | (df["REF"] == 1)]

#     por_cenario = geradores.groupby("scenario").agg(
#         Pg_target=("Pg_target", "sum"),
#         Pg_pred=("Pg_pred", "sum")
#     )

#     por_cenario["erro_relativo_pct"] = (
#         (por_cenario["Pg_pred"] - por_cenario["Pg_target"]).abs()
#         / por_cenario["Pg_target"].abs().replace(0, float("nan"))
#         * 100
#     )

#     erro = por_cenario["erro_relativo_pct"].dropna()

#     print(f"\n===== {nome} =====")
#     print("Erro relativo mediano (%):", erro.median())
#     print("Percentil 95 (%):", erro.quantile(0.95))
#     print("Percentil 99 (%):", erro.quantile(0.99))
#     print("Máximo (%):", erro.max())



# # Comparação da potência reativa total dos geradores
# # IEEE 14 vs IEEE 30

# for nome, df in [("IEEE 14", df14), ("IEEE 30", df30)]:
#     geradores = df[(df["PV"] == 1) | (df["REF"] == 1)]

#     por_cenario = geradores.groupby("scenario").agg(
#         Qg_target_total=("Qg_target", "sum"),
#         Qg_pred_total=("Qg_pred", "sum")
#     )

#     # Erro absoluto na soma da potência reativa
#     por_cenario["erro_total_abs"] = (
#         por_cenario["Qg_pred_total"] -
#         por_cenario["Qg_target_total"]
#     ).abs()

#     # Erro relativo, em percentagem
#     por_cenario["erro_relativo_pct"] = (
#         por_cenario["erro_total_abs"] /
#         por_cenario["Qg_target_total"].abs().replace(0, float("nan"))
#         * 100
#     )

#     print(f"\n===== {nome} =====")
#     print("Erro absoluto médio (Mvar):",
#           por_cenario["erro_total_abs"].mean())
#     print("Erro relativo mediano (%):",
#           por_cenario["erro_relativo_pct"].median())
#     print("Percentil 95 do erro absoluto (Mvar):",
#           por_cenario["erro_total_abs"].quantile(0.95))
#     print("Percentil 99 do erro absoluto (Mvar):",
#           por_cenario["erro_total_abs"].quantile(0.99))
#     print("Erro absoluto máximo (Mvar):",
#           por_cenario["erro_total_abs"].max())



# # Identificar os maiores erros individuais de Qg nos geradores

# for nome, df in [("IEEE 14", df14), ("IEEE 30", df30)]:
#     geradores = df[(df["PV"] == 1) | (df["REF"] == 1)]

#     top_erros = (
#         geradores.sort_values("error_Qg", ascending=False)
#         [["scenario", "bus", "Qg_target", "Qg_pred", "error_Qg"]]
#         .head(10)
#     )

#     print(f"\n===== {nome}: TOP 10 ERROS DE Qg =====")
#     print(top_erros.to_string(index=False))

#     print("\nNúmero de erros acima de 1 Mvar:",
#           (geradores["error_Qg"] > 1).sum())



# # Comparar os erros de Qg perto e longe dos limites operacionais

# for nome, df in [("IEEE 14", df14), ("IEEE 30", df30)]:
#     geradores = df[(df["PV"] == 1) | (df["REF"] == 1)].copy()

#     # Distância da potência reativa de referência ao limite mais próximo
#     margem_min = geradores["Qg_target"] - geradores["Qg_min"]
#     margem_max = geradores["Qg_max"] - geradores["Qg_target"]

#     geradores["margem_Qg"] = margem_min.where(
#         margem_min < margem_max, margem_max
#     )

#     # Considerar perto do limite quando a margem é <= 1 Mvar
#     perto_limite = geradores["margem_Qg"] <= 1
#     longe_limite = geradores["margem_Qg"] > 1

#     print(f"\n===== {nome} =====")

#     print("Número de observações perto do limite:",
#           perto_limite.sum())
#     print("MAE de Qg perto do limite (Mvar):",
#           geradores.loc[perto_limite, "error_Qg"].mean())

#     print("Número de observações longe do limite:",
#           longe_limite.sum())
#     print("MAE de Qg longe do limite (Mvar):",
#           geradores.loc[longe_limite, "error_Qg"].mean())



# # MAE de Qg por barramento gerador

# for nome, df in [("IEEE 14", df14), ("IEEE 30", df30)]:
#     geradores = df[(df["PV"] == 1) | (df["REF"] == 1)]

#     por_bus = (
#         geradores.groupby("bus")
#         .agg(
#             MAE_Qg=("error_Qg", "mean"),
#             P95_Qg=("error_Qg", lambda x: x.quantile(0.95)),
#             MAX_Qg=("error_Qg", "max")
#         )
#         .sort_values("MAE_Qg", ascending=False)
#     )

#     print(f"\n===== {nome}: ERRO DE Qg POR BARRAMENTO =====")
#     print(por_bus.round(4).to_string())



# # Comparar Qg nos barramentos 0 e 1, perto e longe dos limites

# for nome, df in [("IEEE 14", df14), ("IEEE 30", df30)]:
#     print(f"\n===== {nome} =====")

#     geradores = df[
#         ((df["PV"] == 1) | (df["REF"] == 1)) &
#         (df["bus"].isin([0, 1]))
#     ].copy()

#     margem_min = geradores["Qg_target"] - geradores["Qg_min"]
#     margem_max = geradores["Qg_max"] - geradores["Qg_target"]

#     geradores["margem_Qg"] = margem_min.where(
#         margem_min < margem_max, margem_max
#     )

#     geradores["grupo"] = geradores["margem_Qg"].apply(
#         lambda x: "Perto do limite" if x <= 1 else "Longe do limite"
#     )

#     resumo = geradores.groupby(["bus", "grupo"]).agg(
#         observacoes=("error_Qg", "size"),
#         MAE_Qg=("error_Qg", "mean"),
#         P95_Qg=("error_Qg", lambda x: x.quantile(0.95)),
#         MAX_Qg=("error_Qg", "max")
#     )

#     print(resumo.round(4).to_string())



# # Investigar Qg no barramento 1 do IEEE 30
# # Agrupar os valores de referência em intervalos de 5 Mvar

# gerador = df30[
#     ((df30["PV"] == 1) | (df30["REF"] == 1)) &
#     (df30["bus"] == 1)
# ].copy()

# gerador["intervalo_Qg"] = (
#     (gerador["Qg_target"] / 5).round() * 5
# )

# resumo = gerador.groupby("intervalo_Qg").agg(
#     observacoes=("error_Qg", "size"),
#     Qg_medio=("Qg_target", "mean"),
#     MAE_Qg=("error_Qg", "mean"),
#     P95_Qg=("error_Qg", lambda x: x.quantile(0.95)),
#     MAX_Qg=("error_Qg", "max")
# ).sort_index()

# print("\n===== IEEE 30 — BARRAMENTO 1 =====")
# print(resumo.round(3).to_string())



# # Comparar Qg previsto e de referência no barramento 1 do IEEE 30

# for intervalo in [30, 45]:
#     grupo = gerador[
#         gerador["intervalo_Qg"] == intervalo
#     ]

#     print(f"\n===== Intervalo {intervalo} Mvar =====")
#     print("Número de observações:", len(grupo))
#     print("Qg de referência médio:",
#           grupo["Qg_target"].mean())
#     print("Qg previsto médio:",
#           grupo["Qg_pred"].mean())
#     print("Erro assinado médio:",
#           (grupo["Qg_pred"] - grupo["Qg_target"]).mean())
#     print("MAE:", grupo["error_Qg"].mean())



# import matplotlib.pyplot as plt

# plt.figure(figsize=(8, 6))

# plt.scatter(
#     gerador["Qg_target"],
#     gerador["Qg_pred"],
#     s=8,
#     alpha=0.25
# )

# limite_min = min(
#     gerador["Qg_target"].min(),
#     gerador["Qg_pred"].min()
# )
# limite_max = max(
#     gerador["Qg_target"].max(),
#     gerador["Qg_pred"].max()
# )

# plt.plot(
#     [limite_min, limite_max],
#     [limite_min, limite_max],
#     linestyle="--",
#     label="Previsão perfeita"
# )

# plt.xlabel("Qg de referência (Mvar)")
# plt.ylabel("Qg previsto (Mvar)")
# plt.title("IEEE 30 — Barramento 1: referência vs previsão")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()



# # Enviesamento de Qg por intervalo no barramento 1 do IEEE 30

# gerador["erro_assinado"] = (
#     gerador["Qg_pred"] - gerador["Qg_target"]
# )

# gerador["intervalo_Qg"] = (
#     (gerador["Qg_target"] / 5).round() * 5
# )

# resumo_bias = gerador.groupby("intervalo_Qg").agg(
#     observacoes=("erro_assinado", "size"),
#     erro_medio=("erro_assinado", "mean"),
#     MAE=("error_Qg", "mean")
# ).sort_index()

# print("\n===== ENVIESAMENTO DE Qg — IEEE 30, BARRAMENTO 1 =====")
# print(resumo_bias.round(3).to_string())



# # Comparar o enviesamento de Qg no barramento 1 das duas redes

# for nome, df in [("IEEE 14", df14), ("IEEE 30", df30)]:
#     gerador_bus1 = df[
#         ((df["PV"] == 1) | (df["REF"] == 1)) &
#         (df["bus"] == 1)
#     ].copy()

#     gerador_bus1["erro_assinado"] = (
#         gerador_bus1["Qg_pred"] - gerador_bus1["Qg_target"]
#     )

#     gerador_bus1["intervalo_Qg"] = (
#         (gerador_bus1["Qg_target"] / 5).round() * 5
#     )

#     resumo = gerador_bus1.groupby("intervalo_Qg").agg(
#         observacoes=("erro_assinado", "size"),
#         erro_medio=("erro_assinado", "mean"),
#         MAE=("error_Qg", "mean")
#     ).sort_index()

#     print(f"\n===== {nome} — BARRAMENTO 1 =====")
#     print(resumo.round(3).to_string())




# # Comparação global de Qg no barramento 1

# for nome, df in [("IEEE 14", df14), ("IEEE 30", df30)]:
#     dados = df[
#         ((df["PV"] == 1) | (df["REF"] == 1)) &
#         (df["bus"] == 1)
#     ]

#     erro_assinado = dados["Qg_pred"] - dados["Qg_target"]

#     print(f"\n===== {nome} — BARRAMENTO 1 =====")
#     print("Observações:", len(dados))
#     print("Erro assinado médio (Mvar):", erro_assinado.mean())
#     print("MAE (Mvar):", dados["error_Qg"].mean())
#     print("RMSE (Mvar):",
#           (erro_assinado.pow(2).mean()) ** 0.5)



# # Comparação global de Qg no barramento 0

# for nome, df in [("IEEE 14", df14), ("IEEE 30", df30)]:
#     dados = df[
#         ((df["PV"] == 1) | (df["REF"] == 1)) &
#         (df["bus"] == 0)
#     ]

#     erro_assinado = dados["Qg_pred"] - dados["Qg_target"]

#     print(f"\n===== {nome} — BARRAMENTO 0 =====")
#     print("Observações:", len(dados))
#     print("Erro assinado médio (Mvar):", erro_assinado.mean())
#     print("MAE (Mvar):", dados["error_Qg"].mean())
#     print("RMSE (Mvar):",
#           (erro_assinado.pow(2).mean()) ** 0.5)



# # Compensação dos erros de Qg nos barramentos 0 e 1

# for nome, df in [("IEEE 14", df14), ("IEEE 30", df30)]:
#     geradores = df[
#         ((df["PV"] == 1) | (df["REF"] == 1)) &
#         (df["bus"].isin([0, 1]))
#     ].copy()

#     geradores["erro_assinado_Qg"] = (
#         geradores["Qg_pred"] - geradores["Qg_target"]
#     )

#     por_cenario = geradores.groupby("scenario").agg(
#         soma_erros_assinados=("erro_assinado_Qg", "sum"),
#         soma_erros_absolutos=("error_Qg", "sum")
#     )

#     erro_total = por_cenario["soma_erros_assinados"].abs()
#     erros_individuais = por_cenario["soma_erros_absolutos"]

#     compensacao = (
#         1 - erro_total / erros_individuais.replace(0, float("nan"))
#     )

#     print(f"\n===== {nome} =====")
#     print("Erro absoluto médio da soma (Mvar):",
#           erro_total.mean())
#     print("Soma média dos erros absolutos individuais (Mvar):",
#           erros_individuais.mean())
#     print("Compensação média (%):",
#           compensacao.mean() * 100)



# # ============================================================
# # P95 dos erros de Pg e Qg apenas nos barramentos geradores
# # ============================================================

# import numpy as np
# import pandas as pd

# redes = {
#     "IEEE 14-bus": df14,
#     "IEEE 30-bus": df30
# }

# resultados = []

# for nome_rede, df in redes.items():

#     # Selecionar barramentos geradores (PV ou REF)
#     geradores = df[(df["PV"] == 1) | (df["REF"] == 1)].copy()

#     for variavel in ["Pg", "Qg"]:

#         coluna_erro = f"error_{variavel}"

#         # Garantir que o erro é calculado em valor absoluto
#         erros = (geradores[f"{variavel}_pred"] -
#                  geradores[f"{variavel}_target"]).abs()

#         resultados.append({
#             "Rede": nome_rede,
#             "Variável": variavel,
#             "N.º de observações": erros.count(),
#             "MAE": erros.mean(),
#             "RMSE": np.sqrt((erros ** 2).mean()),
#             "P95": erros.quantile(0.95),
#             "P99": erros.quantile(0.99),
#             "Máximo": erros.max()
#         })

# # Criar e apresentar a tabela
# tabela_erros = pd.DataFrame(resultados)

# print("\n=== Erros nas potências dos barramentos geradores ===")
# print(tabela_erros.round(6).to_string(index=False))

# # Guardar os resultados num CSV
# tabela_erros.to_csv(
#     "comparacao_erros_geradores.csv",
#     index=False,
#     encoding="utf-8-sig"
# )

# print("\nTabela guardada em comparacao_erros_geradores.csv")



# # ============================================================
# # Análise dos erros extremos de potência ativa (Pg)
# # ============================================================

# for nome_rede, df in redes.items():

#     geradores = df[(df["PV"] == 1) | (df["REF"] == 1)].copy()

#     geradores["error_Pg"] = (
#         geradores["Pg_pred"] - geradores["Pg_target"]
#     ).abs()

#     print(f"\n=== {nome_rede}: erros de Pg superiores a 1 MW ===")

#     extremos = geradores[geradores["error_Pg"] > 1]

#     print(f"Número de casos: {len(extremos)}")

#     print("\nErros por barramento:")
#     print(
#         extremos.groupby("bus")["error_Pg"]
#         .agg(["count", "mean", "max"])
#         .sort_values("count", ascending=False)
#         .round(4)
#         .to_string()
#     )

#     print("\n10 maiores erros:")
#     print(
#         extremos.sort_values("error_Pg", ascending=False)[
#             ["scenario", "bus", "Pg_target", "Pg_pred", "error_Pg"]
#         ].head(10).to_string(index=False)
#     )



# # ============================================================
# # Compensação dos erros de potência ativa por cenário
# # ============================================================

# for nome_rede, df in redes.items():

#     geradores = df[(df["PV"] == 1) | (df["REF"] == 1)].copy()

#     # Erro assinado: previsão menos valor de referência
#     geradores["erro_assinado_Pg"] = (
#         geradores["Pg_pred"] - geradores["Pg_target"]
#     )

#     # Somar as potências de todos os geradores de cada cenário
#     por_cenario = geradores.groupby("scenario").agg(
#         Pg_target_total=("Pg_target", "sum"),
#         Pg_pred_total=("Pg_pred", "sum"),
#         soma_erros_absolutos=("erro_assinado_Pg",
#                              lambda x: x.abs().sum())
#     )

#     # Erro absoluto entre as somas totais
#     por_cenario["erro_total_Pg"] = (
#         por_cenario["Pg_pred_total"]
#         - por_cenario["Pg_target_total"]
#     ).abs()

#     print(f"\n=== {nome_rede}: erro total de Pg por cenário ===")

#     print(
#         por_cenario[
#             ["erro_total_Pg", "soma_erros_absolutos"]
#         ].quantile([0.50, 0.95, 0.99, 1.00]).round(6)
#     )

#     # Percentagem média de compensação dos erros em cada cenário
#     denominador = por_cenario["soma_erros_absolutos"]

#     compensacao = (
#         1 - por_cenario["erro_total_Pg"] / denominador
#     ).where(denominador > 0).dropna()

#     print(
#         f"Compensação média dos erros: "
#         f"{compensacao.mean() * 100:.2f}%"
#     )



# # ============================================================
# # Identificar os 10 cenários com maior erro total de Pg
# # ============================================================

# for nome_rede, df in redes.items():

#     geradores = df[(df["PV"] == 1) | (df["REF"] == 1)].copy()

#     por_cenario = geradores.groupby("scenario").agg(
#         Pg_target_total=("Pg_target", "sum"),
#         Pg_pred_total=("Pg_pred", "sum")
#     )

#     por_cenario["erro_total_Pg"] = (
#         por_cenario["Pg_pred_total"]
#         - por_cenario["Pg_target_total"]
#     ).abs()

#     piores_cenarios = por_cenario.sort_values(
#         "erro_total_Pg", ascending=False
#     ).head(10)

#     print(f"\n=== {nome_rede}: 10 cenários com maior erro total ===")
#     print(piores_cenarios.round(5).to_string())

#     # Guardar os resultados para análise posterior
#     piores_cenarios.to_csv(
#         f"piores_cenarios_Pg_{nome_rede.replace(' ', '_')}.csv",
#         encoding="utf-8-sig"
#     )



# # ============================================================
# # Analisar os erros individuais nos piores cenários
# # ============================================================

# piores = {
#     "IEEE 14-bus": (df14, 170381),
#     "IEEE 30-bus": (df30, 277123)
# }

# for nome_rede, (df, id_cenario) in piores.items():

#     # Selecionar o cenário e os barramentos geradores
#     dados = df[
#         (df["scenario"] == id_cenario) &
#         ((df["PV"] == 1) | (df["REF"] == 1))
#     ].copy()

#     # Erro assinado e erro absoluto
#     dados["erro_assinado_Pg"] = (
#         dados["Pg_pred"] - dados["Pg_target"]
#     )

#     dados["erro_abs_Pg"] = dados["erro_assinado_Pg"].abs()

#     # Ordenar do maior para o menor erro absoluto
#     dados = dados.sort_values("erro_abs_Pg", ascending=False)

#     print(f"\n=== {nome_rede}: cenário {id_cenario} ===")

#     print(
#         dados[
#             ["bus", "Pg_target", "Pg_pred",
#              "erro_assinado_Pg", "erro_abs_Pg"]
#         ].round(5).to_string(index=False)
#     )

#     print(
#         f"\nSoma dos erros assinados: "
#         f"{dados['erro_assinado_Pg'].sum():.5f} MW"
#     )

#     print(
#         f"Soma dos erros absolutos: "
#         f"{dados['erro_abs_Pg'].sum():.5f} MW"
#     )



# # ============================================================
# # Viés de Pg por barramento gerador
# # ============================================================

# for nome_rede, df in redes.items():

#     geradores = df[
#         (df["PV"] == 1) | (df["REF"] == 1)
#     ].copy()

#     geradores["erro_assinado_Pg"] = (
#         geradores["Pg_pred"] - geradores["Pg_target"]
#     )

#     resumo = geradores.groupby("bus").agg(
#         numero_casos=("erro_assinado_Pg", "count"),
#         erro_medio=("erro_assinado_Pg", "mean"),
#         erro_absoluto_medio=(
#             "erro_assinado_Pg",
#             lambda x: x.abs().mean()
#         ),
#         erro_minimo=("erro_assinado_Pg", "min"),
#         erro_maximo=("erro_assinado_Pg", "max")
#     )

#     print(f"\n=== {nome_rede}: viés de Pg por barramento ===")
#     print(resumo.round(5).to_string())



# import matplotlib.pyplot as plt

# # ============================================================
# # Distribuição dos erros assinados de Pg nos barramentos 0 e 1
# # ============================================================

# fig, axes = plt.subplots(2, 2, figsize=(13, 8))

# redes_plot = [
#     ("IEEE 14-bus", df14),
#     ("IEEE 30-bus", df30)
# ]

# barramentos = [0, 1]

# for coluna, (nome_rede, df) in enumerate(redes_plot):
#     for linha, bus in enumerate(barramentos):

#         dados = df[df["bus"] == bus]

#         erros = dados["Pg_pred"] - dados["Pg_target"]

#         ax = axes[linha, coluna]

#         ax.hist(erros, bins=80, edgecolor="black")
#         ax.axvline(0, linestyle="--", linewidth=1.5)

#         ax.set_title(f"{nome_rede} — Barramento {bus}")
#         ax.set_xlabel("Erro assinado de Pg (MW)")
#         ax.set_ylabel("Número de observações")
#         ax.grid(alpha=0.25)

# plt.tight_layout()
# plt.show()



# # ============================================================
# # Frequência de erros de Pg acima de vários limiares
# # ============================================================

# limiares = [0.1, 0.5, 1.0, 5.0]

# for nome_rede, df in redes.items():

#     print(f"\n=== {nome_rede}: frequência de erros de Pg ===")

#     for bus in [0, 1]:

#         dados = df[df["bus"] == bus]

#         erro_abs = (
#             dados["Pg_pred"] - dados["Pg_target"]
#         ).abs()

#         print(f"\nBarramento {bus} — {len(erro_abs)} cenários")

#         for limite in limiares:

#             n = (erro_abs > limite).sum()
#             percentagem = 100 * n / len(erro_abs)

#             print(
#                 f"Erro > {limite:.1f} MW: "
#                 f"{n} cenários ({percentagem:.3f}%)"
#             )



# # ============================================================
# # Análise dos erros extremos de Pg no IEEE 30-bus
# # ============================================================

# df = df30.copy()

# # Selecionar os barramentos 0 e 1
# dados = df[df["bus"].isin([0, 1])].copy()

# # Erro assinado e erro absoluto
# dados["erro_assinado_Pg"] = (
#     dados["Pg_pred"] - dados["Pg_target"]
# )

# dados["erro_abs_Pg"] = dados["erro_assinado_Pg"].abs()

# # Identificar cenários com erro superior a 5 MW
# extremos = dados[dados["erro_abs_Pg"] > 5].copy()

# print("\n=== Erros de Pg superiores a 5 MW — IEEE 30-bus ===")

# print(
#     extremos[
#         ["scenario", "bus", "Pg_target", "Pg_pred",
#          "erro_assinado_Pg", "erro_abs_Pg"]
#     ].sort_values("erro_abs_Pg", ascending=False)
#     .round(5)
#     .to_string(index=False)
# )

# # Separar os erros dos barramentos 0 e 1 por cenário
# tabela = extremos.pivot(
#     index="scenario",
#     columns="bus",
#     values="erro_assinado_Pg"
# )

# tabela = tabela.rename(columns={
#     0: "erro_bus0_MW",
#     1: "erro_bus1_MW"
# })

# # Incluir todos os cenários em que pelo menos um dos dois
# # barramentos apresenta erro absoluto superior a 5 MW
# tabela["erro_bus0_MW"] = tabela["erro_bus0_MW"].fillna(0)
# tabela["erro_bus1_MW"] = tabela["erro_bus1_MW"].fillna(0)

# tabela["erro_soma_bus0_bus1_MW"] = (
#     tabela["erro_bus0_MW"] + tabela["erro_bus1_MW"]
# )

# tabela["soma_erros_abs_MW"] = (
#     tabela["erro_bus0_MW"].abs()
#     + tabela["erro_bus1_MW"].abs()
# )

# # Verificar se os erros têm sinais opostos
# tabela["sinais_opostos"] = (
#     tabela["erro_bus0_MW"] * tabela["erro_bus1_MW"] < 0
# )

# print("\n=== Compensação entre os barramentos 0 e 1 ===")
# print(tabela.round(5).to_string())

# print("\nNúmero de cenários extremos:", len(tabela))
# print(
#     "Cenários com erros de sinais opostos:",
#     int(tabela["sinais_opostos"].sum())
# )

# # Guardar resultados
# tabela.to_csv(
#     "erros_extremos_Pg_IEEE30.csv",
#     encoding="utf-8-sig"
# )

# print("\nResultados guardados em erros_extremos_Pg_IEEE30.csv")



# # ============================================================
# # Compensação real dos erros nos cenários extremos do IEEE 30
# # ============================================================

# df = df30.copy()

# dados = df[df["bus"].isin([0, 1])].copy()

# dados["erro_assinado_Pg"] = (
#     dados["Pg_pred"] - dados["Pg_target"]
# )

# # Identificar cenários onde pelo menos um dos barramentos
# # tem erro absoluto superior a 5 MW
# extremos = dados[
#     dados["erro_assinado_Pg"].abs() > 5
# ]["scenario"].unique()

# # Recuperar os dois barramentos completos nesses cenários
# dados_extremos = dados[dados["scenario"].isin(extremos)]

# tabela = dados_extremos.pivot(
#     index="scenario",
#     columns="bus",
#     values="erro_assinado_Pg"
# )

# tabela = tabela.rename(columns={
#     0: "erro_bus0_MW",
#     1: "erro_bus1_MW"
# })

# tabela["erro_soma_MW"] = (
#     tabela["erro_bus0_MW"] + tabela["erro_bus1_MW"]
# )

# tabela["soma_erros_abs_MW"] = (
#     tabela["erro_bus0_MW"].abs()
#     + tabela["erro_bus1_MW"].abs()
# )

# tabela["sinais_opostos"] = (
#     tabela["erro_bus0_MW"] * tabela["erro_bus1_MW"] < 0
# )

# print("\n=== Compensação real nos cenários extremos ===")
# print(tabela.round(5).to_string())

# tabela.to_csv(
#     "compensacao_real_Pg_IEEE30.csv",
#     encoding="utf-8-sig"
# )



# # ============================================================
# # Compensação dos erros de Pg nos barramentos 0 e 1
# # em todos os cenários do IEEE 14 e IEEE 30
# # ============================================================

# for nome_rede, df in redes.items():

#     dados = df[df["bus"].isin([0, 1])].copy()

#     dados["erro_assinado"] = (
#         dados["Pg_pred"] - dados["Pg_target"]
#     )

#     erros = dados.pivot(
#         index="scenario",
#         columns="bus",
#         values="erro_assinado"
#     )

#     erros["soma_abs"] = (
#         erros[0].abs() + erros[1].abs()
#     )

#     erros["erro_combinado"] = (
#         erros[0] + erros[1]
#     ).abs()

#     erros["compensacao_pct"] = (
#         100 * (
#             1 - erros["erro_combinado"] / erros["soma_abs"]
#         )
#     ).where(erros["soma_abs"] > 0)

#     print(f"\n=== {nome_rede}: compensação em todos os cenários ===")

#     print(
#         erros[["soma_abs", "erro_combinado", "compensacao_pct"]]
#         .quantile([0.50, 0.95, 0.99, 1.00])
#         .round(4)
#     )

#     print(
#         f"Compensação média: "
#         f"{erros['compensacao_pct'].mean():.2f}%"
#     )



# # ============================================================
# # VERIFICAR ERROS DE Pg NOS RESTANTES GERADORES
# # ============================================================

# for nome_rede, df in redes.items():

#     # Selecionar todos os barramentos geradores
#     geradores = df[
#         (df["PV"] == 1) | (df["REF"] == 1)
#     ].copy()

#     # Calcular o erro assinado de Pg
#     geradores["erro_Pg"] = (
#         geradores["Pg_pred"] - geradores["Pg_target"]
#     )

#     # Excluir os barramentos 0 e 1
#     restantes = geradores[
#         ~geradores["bus"].isin([0, 1])
#     ]

#     # Verificar se existem erros diferentes de zero
#     erros_restantes = restantes[
#         restantes["erro_Pg"].abs() > 1e-6
#     ]

#     print(f"\n{'=' * 55}")
#     print(nome_rede)
#     print("=" * 55)

#     print(
#         "Número de registos dos restantes geradores:",
#         len(restantes)
#     )

#     print(
#         "Registos com erro de Pg superior a 1e-6 MW:",
#         len(erros_restantes)
#     )

#     print(
#         "Número de cenários com pelo menos um erro:",
#         erros_restantes["scenario"].nunique()
#     )

#     if not erros_restantes.empty:
#         print("\nMaiores erros absolutos nos restantes geradores:")

#         print(
#             erros_restantes[
#                 ["scenario", "bus", "Pg_target",
#                  "Pg_pred", "erro_Pg"]
#             ]
#             .assign(
#                 erro_abs=lambda x: x["erro_Pg"].abs()
#             )
#             .sort_values("erro_abs", ascending=False)
#             .head(10)
#             .drop(columns="erro_abs")
#             .to_string(index=False)
#         )
#     else:
#         print(
#             "Todos os restantes geradores têm erro "
#             "inferior ou igual a 1e-6 MW."
#         )



# # ============================================================
# # COMPARAÇÃO DE 3 MODELOS: PBE, STRESS E INTERAÇÃO
# # ============================================================

# import numpy as np
# import pandas as pd
# import statsmodels.formula.api as smf

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     mean_absolute_error,
#     mean_squared_error,
#     r2_score
# )

# print("\n" + "=" * 65)
# print("COMPARAÇÃO DE MODELOS: PBE VS STRESS")
# print("=" * 65)

# # 1. Preparar os dados: uma linha por cenário
# dados_ml = regression_data[
#     ["MAE_active", "PBE", "stress_voltage"]
# ].dropna().copy()

# print(f"Número total de cenários: {len(dados_ml)}")

# # 2. Definir os três modelos
# modelos = {
#     "Modelo 1 - PBE": 
#         "MAE_active ~ PBE",

#     "Modelo 2 - PBE + stress": 
#         "MAE_active ~ PBE + C(stress_voltage)",

#     "Modelo 3 - PBE x stress": 
#         "MAE_active ~ PBE * C(stress_voltage)"
# }

# # 3. Repetir a avaliação com as mesmas divisões
# seeds = [42, 123, 2026, 7, 99]
# resultados = []

# for seed in seeds:

#     treino, teste = train_test_split(
#         dados_ml,
#         test_size=0.20,
#         random_state=seed
#     )

#     for nome, formula in modelos.items():

#         modelo = smf.ols(
#             formula,
#             data=treino
#         ).fit()

#         previsoes = modelo.predict(teste)

#         mae = mean_absolute_error(
#             teste["MAE_active"],
#             previsoes
#         )

#         rmse = np.sqrt(
#             mean_squared_error(
#                 teste["MAE_active"],
#                 previsoes
#             )
#         )

#         r2 = r2_score(
#             teste["MAE_active"],
#             previsoes
#         )

#         resultados.append({
#             "Seed": seed,
#             "Modelo": nome,
#             "MAE": mae,
#             "RMSE": rmse,
#             "R2": r2
#         })

# # 4. Resultados de cada divisão
# resultados_df = pd.DataFrame(resultados)

# print("\n--- Resultados por seed ---")
# print(
#     resultados_df.to_string(
#         index=False,
#         formatters={
#             "MAE": "{:.8f}".format,
#             "RMSE": "{:.8f}".format,
#             "R2": "{:.6f}".format
#         }
#     )
# )

# # 5. Média e desvio-padrão das cinco divisões
# resumo = (
#     resultados_df
#     .groupby("Modelo")
#     .agg(
#         MAE_medio=("MAE", "mean"),
#         MAE_std=("MAE", "std"),
#         RMSE_medio=("RMSE", "mean"),
#         RMSE_std=("RMSE", "std"),
#         R2_medio=("R2", "mean"),
#         R2_std=("R2", "std")
#     )
#     .sort_values("RMSE_medio")
# )

# print("\n--- Resumo final: média +/- desvio-padrão ---")
# print(resumo.to_string(float_format=lambda x: f"{x:.8f}"))

# # 6. Contar em quantas seeds cada modelo obteve
# # o menor RMSE
# vencedores = (
#     resultados_df
#     .loc[
#         resultados_df.groupby("Seed")["RMSE"].idxmin(),
#         "Modelo"
#     ]
#     .value_counts()
# )

# print("\n--- Número de vitórias por RMSE ---")
# for nome in modelos:
#     print(f"{nome}: {vencedores.get(nome, 0)} / {len(seeds)}")

# # 7. Guardar os resultados
# resultados_df.to_csv(
#     "comparacao_modelos_PBE_stress.csv",
#     index=False
# )

# resumo.to_csv(
#     "resumo_modelos_PBE_stress.csv"
# )

# print("\nFicheiros CSV guardados com sucesso.")



# # ============================================================
# # COMPARAÇÃO FINAL: 5 MODELOS PBE / STRESS
# # ============================================================

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     mean_absolute_error,
#     mean_squared_error,
#     r2_score
# )
# import statsmodels.formula.api as smf
# import numpy as np
# import pandas as pd

# # 1. Preparar os dados
# dados_ml = regression_data[
#     ["MAE_active", "PBE", "stress_voltage"]
# ].dropna().copy()

# print("\n" + "=" * 75)
# print("COMPARAÇÃO FINAL DOS MODELOS")
# print("=" * 75)
# print(f"Número de cenários: {len(dados_ml)}")

# # 2. Definir os cinco modelos
# modelos = {
#     "1 - PBE linear":
#         "MAE_active ~ PBE",

#     "2 - PBE + stress":
#         "MAE_active ~ PBE + C(stress_voltage)",

#     "3 - PBE x stress":
#         "MAE_active ~ PBE * C(stress_voltage)",

#     "4 - PBE quadrático":
#         "MAE_active ~ PBE + I(PBE**2)",

#     "5 - Quadrático + stress":
#         "MAE_active ~ (PBE + I(PBE**2)) * C(stress_voltage)"
# }

# # 3. Avaliar todos com as mesmas divisões
# seeds = [42, 123, 2026, 7, 99]
# resultados = []

# for seed in seeds:

#     treino, teste = train_test_split(
#         dados_ml,
#         test_size=0.20,
#         random_state=seed
#     )

#     for nome, formula in modelos.items():

#         modelo = smf.ols(
#             formula,
#             data=treino
#         ).fit()

#         previsoes = modelo.predict(teste)

#         resultados.append({
#             "Seed": seed,
#             "Modelo": nome,
#             "MAE": mean_absolute_error(
#                 teste["MAE_active"],
#                 previsoes
#             ),
#             "RMSE": np.sqrt(
#                 mean_squared_error(
#                     teste["MAE_active"],
#                     previsoes
#                 )
#             ),
#             "R2": r2_score(
#                 teste["MAE_active"],
#                 previsoes
#             )
#         })

# resultados_df = pd.DataFrame(resultados)

# # 4. Resultados individuais por seed
# print("\n--- RESULTADOS POR SEED ---")
# print(
#     resultados_df.to_string(
#         index=False,
#         formatters={
#             "MAE": "{:.8f}".format,
#             "RMSE": "{:.8f}".format,
#             "R2": "{:.6f}".format
#         }
#     )
# )

# # 5. Média e desvio-padrão
# resumo = (
#     resultados_df
#     .groupby("Modelo")
#     .agg(
#         MAE_medio=("MAE", "mean"),
#         MAE_std=("MAE", "std"),
#         RMSE_medio=("RMSE", "mean"),
#         RMSE_std=("RMSE", "std"),
#         R2_medio=("R2", "mean"),
#         R2_std=("R2", "std")
#     )
#     .sort_values("RMSE_medio")
# )

# print("\n--- RESUMO: MÉDIA E DESVIO-PADRÃO ---")
# print(resumo.to_string(float_format=lambda x: f"{x:.8f}"))

# # 6. Número de vitórias por métrica
# print("\n--- VITÓRIAS POR SEED ---")

# for metrica, objetivo in [
#     ("MAE", "min"),
#     ("RMSE", "min"),
#     ("R2", "max")
# ]:

#     if objetivo == "min":
#         indices = resultados_df.groupby("Seed")[metrica].idxmin()
#     else:
#         indices = resultados_df.groupby("Seed")[metrica].idxmax()

#     vencedores = resultados_df.loc[indices, "Modelo"].value_counts()

#     print(f"\nMétrica: {metrica}")
#     for nome in modelos:
#         print(f"{nome}: {vencedores.get(nome, 0)} / {len(seeds)}")

# # 7. Guardar os resultados
# resultados_df.to_csv(
#     "comparacao_5_modelos_por_seed.csv",
#     index=False
# )

# resumo.to_csv(
#     "comparacao_5_modelos_resumo.csv"
# )

# print("\nResultados guardados em CSV.")
