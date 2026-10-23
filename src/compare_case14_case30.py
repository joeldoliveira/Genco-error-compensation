from pathlib import Path
import numpy as np
import pandas as pd

# Executar este script a partir da pasta principal do GENCO Challenge.
# Estrutura esperada:
# outputs/case14_seed42/case14_ieee_predictions.parquet
# outputs/case30_seed42/case30_ieee_predictions.parquet

CASES = {
    "IEEE-14": Path("outputs/case14_seed42/case14_ieee_predictions.parquet"),
    "IEEE-30": Path("outputs/case30_seed42/case30_ieee_predictions.parquet"),
}
OUT = Path("outputs/comparacao_case14_case30")
OUT.mkdir(parents=True, exist_ok=True)


def analyse(name: str, path: Path):
    print(f"\n{'=' * 72}\n{name}: {path}\n{'=' * 72}")
    if not path.exists():
        print(f"FICHEIRO NÃO ENCONTRADO: {path.resolve()}")
        print("Confirma o caminho no dicionário CASES, no início do script.")
        return None

    df = pd.read_parquet(path)
    print(f"Dimensão do ficheiro: {df.shape[0]:,} linhas x {df.shape[1]} colunas")
    print("Colunas:", ", ".join(map(str, df.columns)))

    required = {"scenario", "bus", "Pg_target", "Pg_pred"}
    missing = required - set(df.columns)
    if missing:
        print("Não é possível analisar Pg. Faltam colunas:", sorted(missing))
        return None

    print("Cenários únicos:", df["scenario"].nunique())
    print("Buses únicos:", df["bus"].nunique())
    print("Valores em falta nas colunas principais:")
    print(df[["scenario", "bus", "Pg_target", "Pg_pred"]].isna().sum().to_string())

    # Só entram no cálculo de Pg as linhas em que ambas as potências existem.
    gen = df.dropna(subset=["scenario", "bus", "Pg_target", "Pg_pred"]).copy()
    gen["erro_Pg_MW"] = gen["Pg_pred"] - gen["Pg_target"]
    gen["erro_abs_Pg_MW"] = gen["erro_Pg_MW"].abs()

    # Se houver mais do que uma linha por cenário/bus, agrega primeiro por média.
    by_scenario_bus = gen.groupby(["scenario", "bus"], as_index=False).agg(
        Pg_target=("Pg_target", "mean"),
        Pg_pred=("Pg_pred", "mean"),
        erro_Pg_MW=("erro_Pg_MW", "mean"),
    )
    by_scenario_bus["erro_abs_Pg_MW"] = by_scenario_bus["erro_Pg_MW"].abs()

    bus_summary = by_scenario_bus.groupby("bus").agg(
        n_cenarios=("scenario", "nunique"),
        MAE_Pg_MW=("erro_abs_Pg_MW", "mean"),
        RMSE_Pg_MW=("erro_Pg_MW", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        P95_erro_abs_MW=("erro_abs_Pg_MW", lambda x: x.quantile(0.95)),
        P99_erro_abs_MW=("erro_abs_Pg_MW", lambda x: x.quantile(0.99)),
        max_erro_abs_MW=("erro_abs_Pg_MW", "max"),
        erro_medio_assinado_MW=("erro_Pg_MW", "mean"),
        n_erro_maior_1e_6=("erro_abs_Pg_MW", lambda x: int((x > 1e-6).sum())),
    ).reset_index()
    bus_summary.to_csv(OUT / f"{name.lower().replace('-', '').replace(' ', '_')}_erro_pg_por_bus.csv", index=False)
    print("\nErro de Pg por bus (ordenado por MAE):")
    print(bus_summary.sort_values("MAE_Pg_MW", ascending=False).round(8).to_string(index=False))

    # Soma dos erros assinados dos geradores por cenário: indica o erro líquido total.
    per_scenario = by_scenario_bus.groupby("scenario").agg(
        erro_liquido_total_MW=("erro_Pg_MW", "sum"),
        soma_erros_absolutos_MW=("erro_abs_Pg_MW", "sum"),
        n_buses_geradores=("bus", "nunique"),
    ).reset_index()
    per_scenario["erro_liquido_abs_MW"] = per_scenario["erro_liquido_total_MW"].abs()
    per_scenario["compensacao_pct"] = np.where(
        per_scenario["soma_erros_absolutos_MW"] > 0,
        100 * (1 - per_scenario["erro_liquido_abs_MW"] / per_scenario["soma_erros_absolutos_MW"]),
        np.nan,
    )

    # Compensação apenas entre buses 0 e 1, quando ambos estão presentes.
    pivot = by_scenario_bus.pivot(index="scenario", columns="bus", values="erro_Pg_MW")
    if 0 in pivot.columns and 1 in pivot.columns:
        pair = pivot[[0, 1]].rename(columns={0: "erro_bus0_MW", 1: "erro_bus1_MW"}).copy()
        pair[["erro_bus0_MW", "erro_bus1_MW"]] = pair[["erro_bus0_MW", "erro_bus1_MW"]].fillna(0)
        pair["erro_liquido_bus0_bus1_MW"] = pair["erro_bus0_MW"] + pair["erro_bus1_MW"]
        pair["soma_abs_bus0_bus1_MW"] = pair["erro_bus0_MW"].abs() + pair["erro_bus1_MW"].abs()
        pair["compensacao_bus0_bus1_pct"] = np.where(
            pair["soma_abs_bus0_bus1_MW"] > 0,
            100 * (1 - pair["erro_liquido_bus0_bus1_MW"].abs() / pair["soma_abs_bus0_bus1_MW"]),
            np.nan,
        )
        pair["erros_sinais_opostos"] = pair["erro_bus0_MW"] * pair["erro_bus1_MW"] < 0
        pair.reset_index().to_csv(OUT / f"{name.lower().replace('-', '').replace(' ', '_')}_compensacao_buses_0_1.csv", index=False)
        pair_summary = {
            "sistema": name,
            "cenarios_com_bus0_e_bus1": int(len(pair)),
            "percentagem_cenarios_sinais_opostos": 100 * pair["erros_sinais_opostos"].mean(),
            "compensacao_mediana_pct": pair["compensacao_bus0_bus1_pct"].median(),
            "compensacao_media_pct": pair["compensacao_bus0_bus1_pct"].mean(),
            "compensacao_p95_pct": pair["compensacao_bus0_bus1_pct"].quantile(0.95),
        }
    else:
        pair_summary = {"sistema": name, "cenarios_com_bus0_e_bus1": 0}

    summary = {
        "sistema": name,
        "ficheiro": str(path),
        "linhas_originais": len(df),
        "cenarios": int(df["scenario"].nunique()),
        "buses_unicos": int(df["bus"].nunique()),
        "linhas_validas_Pg": len(gen),
        "buses_com_Pg": int(by_scenario_bus["bus"].nunique()),
        "MAE_Pg_por_linha_MW": float(gen["erro_abs_Pg_MW"].mean()),
        "RMSE_Pg_por_linha_MW": float(np.sqrt(np.mean(np.square(gen["erro_Pg_MW"])))),
        "MAE_erro_liquido_total_por_cenario_MW": float(per_scenario["erro_liquido_abs_MW"].mean()),
        "P95_erro_liquido_total_MW": float(per_scenario["erro_liquido_abs_MW"].quantile(0.95)),
        "compensacao_media_total_pct": float(per_scenario["compensacao_pct"].mean()),
        **pair_summary,
    }
    per_scenario.to_csv(OUT / f"{name.lower().replace('-', '').replace(' ', '_')}_compensacao_total_por_cenario.csv", index=False)
    return summary


summaries = []
for name, path in CASES.items():
    result = analyse(name, path)
    if result is not None:
        summaries.append(result)

if summaries:
    comparison = pd.DataFrame(summaries)
    comparison.to_csv(OUT / "resumo_comparacao_case14_case30.csv", index=False)
    print(f"\n{'=' * 72}\nRESUMO COMPARATIVO\n{'=' * 72}")
    print(comparison.round(6).to_string(index=False))
    print(f"\nFicheiros de resultados guardados em: {OUT.resolve()}")
else:
    print("\nNão foi possível analisar nenhum ficheiro. Confirma os caminhos e as colunas.")
