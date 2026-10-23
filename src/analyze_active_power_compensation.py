from pathlib import Path
import numpy as np
import pandas as pd

# Executar na pasta principal do projeto GENCO Challenge.
CASES = {
    "IEEE-14": Path("outputs/case14_seed42/case14_ieee_predictions.parquet"),
    "IEEE-30": Path("outputs/case30_seed42/case30_ieee_predictions.parquet"),
}
OUT = Path("outputs/analise_compensacao_detalhada")
OUT.mkdir(parents=True, exist_ok=True)


def slug(s):
    return s.lower().replace("-", "").replace(" ", "_")


def analyse(name, path):
    print(f"\n{'=' * 78}\n{name}: {path}\n{'=' * 78}")
    if not path.exists():
        print(f"ERRO: não encontrei {path.resolve()}")
        return None

    df = pd.read_parquet(path)
    required = {"scenario", "bus", "Pg_target", "Pg_pred"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERRO: faltam colunas obrigatórias: {sorted(missing)}")
        return None

    data = df.dropna(subset=["scenario", "bus", "Pg_target", "Pg_pred"]).copy()
    data["erro_Pg_MW"] = data["Pg_pred"] - data["Pg_target"]
    # Uma linha por cenário/bus (a média evita duplicados inesperados).
    data = data.groupby(["scenario", "bus"], as_index=False).agg(erro_Pg_MW=("erro_Pg_MW", "mean"))
    pivot = data.pivot(index="scenario", columns="bus", values="erro_Pg_MW").sort_index().fillna(0)

    if 0 not in pivot.columns or 1 not in pivot.columns:
        print("ERRO: não encontrei os buses 0 e 1 no ficheiro.")
        return None

    e0 = pivot[0]
    e1 = pivot[1]
    pair = pd.DataFrame({"scenario": pivot.index, "erro_bus0_MW": e0.values, "erro_bus1_MW": e1.values})
    pair["erro_liquido_0_1_MW"] = pair["erro_bus0_MW"] + pair["erro_bus1_MW"]
    pair["erro_liquido_abs_0_1_MW"] = pair["erro_liquido_0_1_MW"].abs()
    pair["soma_erros_abs_0_1_MW"] = pair["erro_bus0_MW"].abs() + pair["erro_bus1_MW"].abs()
    pair["compensacao_0_1_pct"] = np.where(
        pair["soma_erros_abs_0_1_MW"] > 0,
        100 * (1 - pair["erro_liquido_abs_0_1_MW"] / pair["soma_erros_abs_0_1_MW"]),
        np.nan,
    )
    pair["sinais_opostos"] = pair["erro_bus0_MW"] * pair["erro_bus1_MW"] < 0
    pair["correlacao_contributiva"] = np.where(pair["sinais_opostos"], "opostos", "iguais_ou_zero")

    # Erro líquido e compensação considerando todos os buses por cenário.
    total = data.groupby("scenario").agg(
        erro_liquido_total_MW=("erro_Pg_MW", "sum"),
        soma_erros_absolutos_MW=("erro_Pg_MW", lambda x: x.abs().sum()),
    ).reset_index()
    total["erro_liquido_abs_total_MW"] = total["erro_liquido_total_MW"].abs()
    total["compensacao_total_pct"] = np.where(
        total["soma_erros_absolutos_MW"] > 0,
        100 * (1 - total["erro_liquido_abs_total_MW"] / total["soma_erros_absolutos_MW"]),
        np.nan,
    )

    # Junta as métricas cenário a cenário para facilitar a inspeção.
    scenarios = total.merge(pair, on="scenario", how="left")
    scenarios.to_csv(OUT / f"{slug(name)}_todos_cenarios.csv", index=False)
    pair.to_csv(OUT / f"{slug(name)}_erros_buses_0_1.csv", index=False)

    # Os piores 1%: menor compensação (primeiro) e maior erro líquido (segundo).
    n_worst = max(1, int(np.ceil(len(scenarios) * 0.01)))
    worst_comp_total = scenarios.sort_values(["compensacao_total_pct", "erro_liquido_abs_total_MW"], ascending=[True, False]).head(n_worst)
    worst_comp_pair = scenarios.sort_values(["compensacao_0_1_pct", "erro_liquido_abs_0_1_MW"], ascending=[True, False]).head(n_worst)
    worst_net_total = scenarios.sort_values("erro_liquido_abs_total_MW", ascending=False).head(n_worst)
    worst_net_pair = scenarios.sort_values("erro_liquido_abs_0_1_MW", ascending=False).head(n_worst)
    worst_comp_total.to_csv(OUT / f"{slug(name)}_piores_1pct_compensacao_total.csv", index=False)
    worst_comp_pair.to_csv(OUT / f"{slug(name)}_piores_1pct_compensacao_buses_0_1.csv", index=False)
    worst_net_total.to_csv(OUT / f"{slug(name)}_piores_1pct_erro_liquido_total.csv", index=False)
    worst_net_pair.to_csv(OUT / f"{slug(name)}_piores_1pct_erro_liquido_buses_0_1.csv", index=False)

    # Correlação entre os erros dos dois buses. A correlação negativa é consistente
    # com erros que variam em sentidos opostos, mas não é por si só prova de causalidade.
    pearson = e0.corr(e1, method="pearson")
    spearman = e0.corr(e1, method="spearman")
    summary = {
        "sistema": name,
        "cenarios": int(len(scenarios)),
        "correlacao_pearson_bus0_bus1": float(pearson),
        "correlacao_spearman_bus0_bus1": float(spearman),
        "percentagem_sinais_opostos_bus0_bus1": float(100 * pair["sinais_opostos"].mean()),
        "compensacao_mediana_bus0_bus1_pct": float(pair["compensacao_0_1_pct"].median()),
        "compensacao_media_bus0_bus1_pct": float(pair["compensacao_0_1_pct"].mean()),
        "p01_compensacao_bus0_bus1_pct": float(pair["compensacao_0_1_pct"].quantile(0.01)),
        "p01_compensacao_total_pct": float(total["compensacao_total_pct"].quantile(0.01)),
        "maior_erro_liquido_total_MW": float(total["erro_liquido_abs_total_MW"].max()),
        "p95_erro_liquido_total_MW": float(total["erro_liquido_abs_total_MW"].quantile(0.95)),
        "maior_erro_liquido_buses_0_1_MW": float(pair["erro_liquido_abs_0_1_MW"].max()),
        "n_cenarios_no_pior_1pct": int(n_worst),
    }

    print(f"Cenários: {len(scenarios):,}")
    print(f"Correlação Pearson (erros buses 0 e 1): {pearson:.5f}")
    print(f"Correlação Spearman (erros buses 0 e 1): {spearman:.5f}")
    print(f"Sinais opostos buses 0 e 1: {summary['percentagem_sinais_opostos_bus0_bus1']:.2f}%")
    print(f"Compensação média/mediana buses 0 e 1: {summary['compensacao_media_bus0_bus1_pct']:.2f}% / {summary['compensacao_mediana_bus0_bus1_pct']:.2f}%")
    print(f"Percentil 1 da compensação total: {summary['p01_compensacao_total_pct']:.2f}%")
    print(f"Maior erro líquido total: {summary['maior_erro_liquido_total_MW']:.6f} MW")
    print(f"\nPiores {n_worst} cenários (1%) por compensação total:")
    print(worst_comp_total[["scenario", "compensacao_total_pct", "erro_liquido_abs_total_MW", "erro_bus0_MW", "erro_bus1_MW"]].head(10).round(6).to_string(index=False))
    print(f"\nMaiores erros líquidos totais (top 10):")
    print(worst_net_total[["scenario", "compensacao_total_pct", "erro_liquido_abs_total_MW", "erro_bus0_MW", "erro_bus1_MW"]].head(10).round(6).to_string(index=False))
    return summary


summaries = []
for name, path in CASES.items():
    result = analyse(name, path)
    if result:
        summaries.append(result)

if summaries:
    result_df = pd.DataFrame(summaries)
    result_df.to_csv(OUT / "resumo_compensacao_detalhada.csv", index=False)
    print(f"\n{'=' * 78}\nRESUMO FINAL\n{'=' * 78}")
    print(result_df.round(6).to_string(index=False))
    print(f"\nResultados guardados em: {OUT.resolve()}")
else:
    print("\nNão foi possível analisar os ficheiros. Confirma os caminhos e a instalação de pyarrow.")
