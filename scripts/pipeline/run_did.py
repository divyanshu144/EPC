#!/usr/bin/env python3
"""
Fixed-effects models, heterogeneity, and clustering analyses
aligned with the dissertation report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from linearmodels.panel import PanelOLS
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

from config import ROOT
from scripts.pipeline.policy import POLICY_ORDER


DATA_PATH = ROOT / "data" / "processed" / "ew_epc_core_clean_sw" / "ew_epc_core_clean_sw_imputed.csv"
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"

sns.set(style="whitegrid")


BASE_RHS = (
    "C(POLICY_PERIOD)"
    " + C(PROPERTY_TYPE)"
    " + C(AGE_BAND_STD)"
    " + C(TENURE_STD)"
    " + C(MAIN_FUEL_STD)"
    " + LOG_FLOOR_AREA"
    " + EntityEffects"
)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def prepare_panel(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    if "BELOW_C_INT" not in df.columns:
        df["BELOW_C_INT"] = df["BELOW_C"].astype(int)
    df["POLICY_PERIOD"] = pd.Categorical(
        df["POLICY_PERIOD"], categories=POLICY_ORDER, ordered=True
    )
    df = df.set_index(["LOCAL_AUTHORITY", "YEAR"]).sort_index()
    return df


def fit_fe(df_panel: pd.DataFrame, dep_var: str):
    formula = f"{dep_var} ~ 1 + {BASE_RHS}"
    clusters = pd.Series(
        df_panel.index.get_level_values("LOCAL_AUTHORITY"),
        index=df_panel.index,
        name="LOCAL_AUTHORITY",
    )
    model = PanelOLS.from_formula(
        formula, data=df_panel, drop_absorbed=True, check_rank=False
    )
    return model.fit(cov_type="clustered", clusters=clusters)


def extract_policy_effects(res, prefix="C(POLICY_PERIOD)"):
    params = res.params.filter(like=prefix)
    se = res.std_errors.filter(like=prefix)
    out = pd.concat([params, se], axis=1).reset_index()
    out.columns = ["term", "coef", "se"]
    out["lower_95"] = out["coef"] - 1.96 * out["se"]
    out["upper_95"] = out["coef"] + 1.96 * out["se"]
    def _term_to_period(term: str) -> str:
        # linearmodels terms typically look like: C(POLICY_PERIOD)[T.MEES]
        if "[" in term and "]" in term:
            inner = term.split("[", 1)[1].rsplit("]", 1)[0]
            return inner.replace("T.", "")
        return term

    out["policy_period"] = out["term"].apply(_term_to_period)
    out = out.drop(columns=["term"]).sort_values("policy_period")
    return out


def plot_policy_effects(df_eff: pd.DataFrame, title: str, x_label: str, out_path: Path):
    df_plot = df_eff.sort_values("policy_period")
    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.errorbar(
        df_plot["coef"],
        df_plot["policy_period"],
        xerr=[
            df_plot["coef"] - df_plot["lower_95"],
            df_plot["upper_95"] - df_plot["coef"],
        ],
        fmt="o",
        color="#2C3E50",
        ecolor="#7F8C8D",
        capsize=3,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Policy period")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_dumbbell_policy(df_a, df_b, label_a, label_b, title, x_label, out_path: Path):
    df = (
        df_a[["policy_period", "coef"]]
        .merge(df_b[["policy_period", "coef"]], on="policy_period", suffixes=("_a", "_b"))
        .sort_values("policy_period")
    )
    y = range(len(df))
    plt.figure(figsize=(7, 4))
    ax = plt.gca()
    ax.hlines(
        y=y,
        xmin=df["coef_a"],
        xmax=df["coef_b"],
        color="grey",
        alpha=0.7,
        linewidth=2,
    )
    ax.scatter(df["coef_a"], y, s=50, label=label_a, zorder=3)
    ax.scatter(df["coef_b"], y, s=50, marker="s", label=label_b, zorder=3)
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["policy_period"])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Policy period")
    ax.legend(frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def clustering_from_random_effects(df_panel: pd.DataFrame, out_prefix: str, sample_n: int = 200_000):
    # Random-effects model with random slopes over policy periods (sampled for tractability)
    df = df_panel.reset_index().copy()
    if len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)

    formula = (
        "CURRENT_ENERGY_EFFICIENCY ~ "
        "C(POLICY_PERIOD) + C(PROPERTY_TYPE) + C(TENURE_STD) + "
        "C(AGE_BAND_STD) + LOG_FLOOR_AREA"
    )
    md = sm.MixedLM.from_formula(
        formula,
        groups="LOCAL_AUTHORITY",
        re_formula="C(POLICY_PERIOD)",
        data=df,
    )
    result = md.fit(reml=True, method="lbfgs", maxiter=200, disp=False)

    re_dict = result.random_effects
    re_df = pd.DataFrame(re_dict).T
    if "LOCAL_AUTHORITY" not in re_df.columns:
        re_df.index.name = "LOCAL_AUTHORITY"
        re_df = re_df.reset_index()
    else:
        re_df.index.name = "LOCAL_AUTHORITY_ID"
        re_df = re_df.reset_index()

    # PCA + clustering
    slope_cols = re_df.select_dtypes(include=[np.number]).columns.tolist()
    X = re_df[slope_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(X_scaled)
    k = 3
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = km.fit_predict(comps)
    re_df["cluster"] = clusters

    sil_score = silhouette_score(comps, clusters)
    sil_vals = silhouette_samples(comps, clusters)

    # PCA scatter
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=comps[:, 0], y=comps[:, 1], hue=clusters, palette="viridis")
    plt.title("PCA of Local-Authority Random Effects (k=3)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / f"{out_prefix}_pca.png", dpi=200)
    plt.close()

    # Silhouette plot
    fig, ax = plt.subplots(figsize=(7, 5))
    y_lower = 10
    for i in range(k):
        ith = np.sort(sil_vals[clusters == i])
        size_i = ith.shape[0]
        y_upper = y_lower + size_i
        color = plt.cm.nipy_spectral(float(i) / k)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith, facecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_i, str(i))
        y_lower = y_upper + 10
    ax.axvline(x=sil_score, color="red", linestyle="--")
    ax.set_title("Silhouette scores (k=3)")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{out_prefix}_silhouette.png", dpi=200)
    plt.close()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    re_df.to_csv(TABLE_DIR / f"{out_prefix}_clusters.csv", index=False)


def main() -> None:
    df = load_data()
    df_panel = prepare_panel(df)

    # Main FE models
    res_epc = fit_fe(df_panel, "CURRENT_ENERGY_EFFICIENCY")
    res_co2 = fit_fe(df_panel, "CO2_EMISSIONS_CURRENT")
    res_belowc = fit_fe(df_panel, "BELOW_C_INT")

    epc_eff = extract_policy_effects(res_epc)
    co2_eff = extract_policy_effects(res_co2)
    belowc_eff = extract_policy_effects(res_belowc)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    epc_eff.to_csv(TABLE_DIR / "policy_effects_epc.csv", index=False)
    co2_eff.to_csv(TABLE_DIR / "policy_effects_co2.csv", index=False)
    belowc_eff.to_csv(TABLE_DIR / "policy_effects_below_c.csv", index=False)

    plot_policy_effects(
        co2_eff,
        title="Policy-period effects on CO2 emissions (vs Pre-GreenDeal)",
        x_label="Change in CO2 emissions (tonnes/year)",
        out_path=FIG_DIR / "figure_6_1_policy_co2.png",
    )
    plot_policy_effects(
        belowc_eff,
        title="Policy-period effects on probability dwelling is below EPC C",
        x_label="Change in probability (vs Pre-GreenDeal)",
        out_path=FIG_DIR / "figure_6_2_policy_below_c.png",
    )

    # Heterogeneity: mains gas vs non-gas
    df_gas = df_panel[df_panel["HAS_MAINS_GAS"]].copy()
    df_nongas = df_panel[~df_panel["HAS_MAINS_GAS"]].copy()

    res_epc_gas = fit_fe(df_gas, "CURRENT_ENERGY_EFFICIENCY")
    res_epc_nongas = fit_fe(df_nongas, "CURRENT_ENERGY_EFFICIENCY")
    epc_gas_eff = extract_policy_effects(res_epc_gas)
    epc_nongas_eff = extract_policy_effects(res_epc_nongas)

    plot_dumbbell_policy(
        epc_gas_eff,
        epc_nongas_eff,
        label_a="Mains gas",
        label_b="Non-gas",
        title="Policy-period effects on EPC efficiency: mains gas vs non-gas",
        x_label="Change in EPC score (vs Pre-GreenDeal)",
        out_path=FIG_DIR / "figure_6_3_gas_vs_nongas.png",
    )

    # Heterogeneity: owner-occupied vs private rented
    df_owner = df_panel[df_panel["TENURE_STD"] == "Owner-occupied"].copy()
    df_prs = df_panel[df_panel["TENURE_STD"] == "Rented (private)"].copy()

    res_co2_owner = fit_fe(df_owner, "CO2_EMISSIONS_CURRENT")
    res_co2_prs = fit_fe(df_prs, "CO2_EMISSIONS_CURRENT")
    co2_owner_eff = extract_policy_effects(res_co2_owner)
    co2_prs_eff = extract_policy_effects(res_co2_prs)

    plot_dumbbell_policy(
        co2_owner_eff,
        co2_prs_eff,
        label_a="Owner-occupied",
        label_b="Private rented",
        title="Policy-period effects on CO2 emissions: owners vs private renters",
        x_label="Change in CO2 emissions (vs Pre-GreenDeal)",
        out_path=FIG_DIR / "figure_6_4_owner_vs_prs_co2.png",
    )

    # Clustering of local authorities
    clustering_from_random_effects(df_panel, out_prefix="figure_4_1_la_clusters")

    print("✓ FE models and clustering completed.")


if __name__ == "__main__":
    main()
