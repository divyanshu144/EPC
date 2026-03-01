#!/usr/bin/env python3
"""
Fixed-effects models, heterogeneity, and clustering analyses
aligned with the dissertation report.

Inference note — few clusters
------------------------------
The South West contains only 14 local authorities. Standard clustered standard
errors (clustered SEs) rely on asymptotic theory that requires a large number
of clusters (rule of thumb: ≥30). With 14 clusters the clustered SEs are
likely downward-biased, making coefficients appear more significant than they
are.

To address this, every FE model is estimated with *both*:
  1. Conventional clustered SEs (cov_type='clustered') — reported for
     comparability with prior literature.
  2. Wild cluster bootstrap p-values (WCB) — a small-sample correction that
     is valid even with very few clusters (Cameron, Gelbach & Miller 2008).
     We use the Rademacher weights variant with B=999 bootstrap replications.

WCB is implemented via a manual loop (linearmodels does not expose it natively)
and reported alongside clustered-SE confidence intervals in the output tables.
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

from config import ROOT, REGISTRY_PATH
from scripts.pipeline.policy import POLICY_ORDER
from ew_housing_energy_impact.registry import register_artifact


DATA_PATH = ROOT / "data" / "processed" / "ew_epc_core_clean_sw" / "ew_epc_core_clean_sw_imputed.csv"
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"

sns.set(style="whitegrid")


# Shared structural controls used across all model specifications.
# NEW_BUILD_SHARE controls for compositional shift: years with many new-build
# EPCs mechanically raise the LA-year mean EPC score independent of retrofit
# policy. Without this control, policy-period coefficients absorb the new-build
# effect and are upward-biased in efficiency models.
_CONTROLS = (
    " + C(PROPERTY_TYPE)"
    " + C(AGE_BAND_STD)"
    " + C(TENURE_STD)"
    " + C(MAIN_FUEL_STD)"
    " + LOG_FLOOR_AREA"
    " + NEW_BUILD_SHARE"
    " + EntityEffects"
)

# Main specification: five policy-period dummies (Pre-GreenDeal = reference).
BASE_RHS = "C(POLICY_PERIOD)" + _CONTROLS

# Event-study specification: year dummies (2008 = reference, dropped by patsy).
# Produces one coefficient per year, showing exactly when trends shift.
EVENT_STUDY_RHS = "C(YEAR)" + _CONTROLS

# Policy-period boundary years for event-study annotation lines.
# Each value is the LAST year of the preceding period.
_PERIOD_BREAKS: list[tuple[int, str]] = [
    (2012, "Pre-GreenDeal\n/ GreenDeal"),
    (2015, "GreenDeal\n/ ECO2"),
    (2018, "ECO2\n/ MEES"),
    (2020, "MEES\n/ Post-Strategy"),
]


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


def wild_cluster_bootstrap_pvalues(
    df_panel: pd.DataFrame,
    dep_var: str,
    B: int = 999,
    seed: int = 42,
) -> pd.Series:
    """Wild cluster bootstrap p-values (Rademacher weights).

    Valid for small numbers of clusters (Cameron, Gelbach & Miller 2008).
    Returns a Series of p-values indexed by parameter name, for terms
    containing 'POLICY_PERIOD'.

    Steps:
      1. Fit the FE model once to get residuals and fitted values.
      2. In each bootstrap iteration, multiply residuals by a cluster-level
         Rademacher weight (+1 or -1, one per LA), refit on the perturbed
         outcome, and record the t-statistics.
      3. The WCB p-value for each coefficient is the share of bootstrap
         t-statistics whose absolute value exceeds the observed |t|.
    """
    rng = np.random.default_rng(seed)
    las = df_panel.index.get_level_values("LOCAL_AUTHORITY")
    unique_las = las.unique()
    n_clusters = len(unique_las)

    # Fit baseline model
    res = fit_fe(df_panel, dep_var)
    residuals = res.resids
    fitted = res.fitted_values

    # Observed t-statistics for policy period terms
    policy_terms = [p for p in res.params.index if "POLICY_PERIOD" in p]
    t_obs = res.tstats[policy_terms].abs()

    t_boot = {term: [] for term in policy_terms}

    for _ in range(B):
        # Draw one Rademacher weight per cluster
        weights = rng.choice([-1.0, 1.0], size=n_clusters)
        la_weight = pd.Series(
            {la: w for la, w in zip(unique_las, weights)}, name="weight"
        )
        obs_weights = las.map(la_weight)

        # Perturbed outcome = fitted + weight * residual
        y_star = fitted + obs_weights.values * residuals.values
        y_star = pd.Series(y_star, index=df_panel.index, name=dep_var)

        df_boot = df_panel.copy()
        df_boot[dep_var] = y_star

        try:
            res_b = fit_fe(df_boot, dep_var)
            for term in policy_terms:
                if term in res_b.tstats.index:
                    t_boot[term].append(abs(res_b.tstats[term]))
        except Exception:
            continue

    # p-value = proportion of |t_boot| >= |t_obs|
    pvals = {}
    for term in policy_terms:
        boot_arr = np.array(t_boot[term])
        if len(boot_arr) == 0:
            pvals[term] = np.nan
        else:
            pvals[term] = (boot_arr >= t_obs[term]).mean()

    return pd.Series(pvals, name="wcb_pvalue")


def extract_policy_effects(res, wcb_pvalues: pd.Series | None = None, prefix="C(POLICY_PERIOD)"):
    """Extract policy-period coefficients, clustered SEs, 95% CIs, and WCB p-values."""
    params = res.params.filter(like=prefix)
    se = res.std_errors.filter(like=prefix)
    clustered_pvals = res.pvalues.filter(like=prefix)
    out = pd.concat([params, se, clustered_pvals], axis=1).reset_index()
    out.columns = ["term", "coef", "se", "clustered_pval"]
    out["lower_95"] = out["coef"] - 1.96 * out["se"]
    out["upper_95"] = out["coef"] + 1.96 * out["se"]

    def _term_to_period(term: str) -> str:
        if "[" in term and "]" in term:
            inner = term.split("[", 1)[1].rsplit("]", 1)[0]
            return inner.replace("T.", "")
        return term

    out["policy_period"] = out["term"].apply(_term_to_period)

    if wcb_pvalues is not None:
        # Map WCB p-values by term name
        out["wcb_pval"] = out["term"].map(wcb_pvalues)
    else:
        out["wcb_pval"] = np.nan

    out = out.drop(columns=["term"]).sort_values("policy_period")
    return out


# ---------------------------------------------------------------------------
# Event-study: year-by-year coefficients
# ---------------------------------------------------------------------------

def fit_event_study(df_panel: pd.DataFrame, dep_var: str):
    """Fit the event-study (year-dummy) FE model.

    Replaces the five broad policy-period dummies with a full set of year
    indicators (2009–2025), using 2008 as the reference year (omitted by
    patsy automatically as the first level of C(YEAR)).

    The structural controls (property type, age band, tenure, fuel, log floor
    area, new-build share, LA fixed effects) are identical to the main spec so
    results are directly comparable.

    Returns the fitted PanelOLS result object.
    """
    formula = f"{dep_var} ~ 1 + {EVENT_STUDY_RHS}"
    clusters = pd.Series(
        df_panel.index.get_level_values("LOCAL_AUTHORITY"),
        index=df_panel.index,
        name="LOCAL_AUTHORITY",
    )
    model = PanelOLS.from_formula(
        formula, data=df_panel, drop_absorbed=True, check_rank=False
    )
    return model.fit(cov_type="clustered", clusters=clusters)


def extract_event_study(res) -> pd.DataFrame:
    """Extract year coefficients and 95% CIs from an event-study result.

    Also prepends the reference year (2008) with coefficient = 0 so the plot
    anchors correctly at the baseline.

    Returns a DataFrame with columns: year, coef, se, lower_95, upper_95.
    """
    prefix = "C(YEAR)"
    params = res.params.filter(like=prefix)
    se = res.std_errors.filter(like=prefix)

    out = pd.concat([params, se], axis=1).reset_index()
    out.columns = ["term", "coef", "se"]
    out["lower_95"] = out["coef"] - 1.96 * out["se"]
    out["upper_95"] = out["coef"] + 1.96 * out["se"]

    def _term_to_year(term: str) -> int:
        # Term looks like: C(YEAR)[T.2013]
        if "[" in term and "]" in term:
            inner = term.split("[", 1)[1].rsplit("]", 1)[0]
            return int(inner.replace("T.", ""))
        return int(term)

    out["year"] = out["term"].apply(_term_to_year)
    out = out.drop(columns=["term"]).sort_values("year").reset_index(drop=True)

    # Prepend reference year (2008) at zero
    ref_row = pd.DataFrame([{
        "year": 2008, "coef": 0.0, "se": 0.0, "lower_95": 0.0, "upper_95": 0.0
    }])
    out = pd.concat([ref_row, out], ignore_index=True).sort_values("year").reset_index(drop=True)
    return out


def plot_event_study(
    df_es: pd.DataFrame,
    title: str,
    y_label: str,
    out_path: Path,
) -> None:
    """Plot event-study year-by-year coefficients with policy-period annotations.

    Layout
    ------
    - Filled dots connected by a thin line show point estimates.
    - Shaded ribbon shows the 95% CI band.
    - Vertical dashed lines mark policy period transitions.
    - A horizontal line at y=0 marks the 2008 baseline.
    - Period labels are placed just above the x-axis between boundary lines.

    Args:
        df_es:    Output of extract_event_study() — columns year, coef, lower_95, upper_95.
        title:    Plot title.
        y_label:  Y-axis label (e.g. 'Change in EPC SAP score vs 2008').
        out_path: Where to save the figure.
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    years = df_es["year"].values
    coefs = df_es["coef"].values
    lo = df_es["lower_95"].values
    hi = df_es["upper_95"].values

    # CI ribbon
    ax.fill_between(years, lo, hi, alpha=0.18, color="#2C7BB6", label="95% CI")

    # Point estimates connected by line
    ax.plot(years, coefs, color="#2C7BB6", linewidth=1.4, zorder=3)
    ax.scatter(years, coefs, color="#2C7BB6", s=40, zorder=4)

    # Baseline
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    # Policy-period boundary lines + period labels
    period_labels = [
        (2008, 2012, "Pre-GreenDeal"),
        (2013, 2015, "GreenDeal\n/ ECO1"),
        (2016, 2018, "ECO2"),
        (2019, 2020, "MEES"),
        (2021, 2025, "Post-\nStrategy"),
    ]
    y_label_pos = ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else lo.min()

    for break_year, label in _PERIOD_BREAKS:
        ax.axvline(
            break_year + 0.5,
            color="grey",
            linewidth=0.9,
            linestyle=":",
            alpha=0.7,
        )

    # Period span labels (centred between boundaries)
    boundaries = [2007.5] + [b + 0.5 for b, _ in _PERIOD_BREAKS] + [2025.5]
    period_names = [
        "Pre-GreenDeal",
        "GreenDeal / ECO1",
        "ECO2",
        "MEES",
        "Post-Strategy",
    ]
    y_range = hi.max() - lo.min()
    label_y = lo.min() - 0.06 * y_range

    for i, name in enumerate(period_names):
        mid_x = (boundaries[i] + boundaries[i + 1]) / 2
        ax.text(
            mid_x,
            label_y,
            name,
            ha="center",
            va="top",
            fontsize=7.5,
            color="#555555",
            clip_on=False,
        )

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_xticks(range(int(years.min()), int(years.max()) + 1, 2))
    ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


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
    print("Fitting FE models...")
    res_epc = fit_fe(df_panel, "CURRENT_ENERGY_EFFICIENCY")
    res_co2 = fit_fe(df_panel, "CO2_EMISSIONS_CURRENT")
    res_belowc = fit_fe(df_panel, "BELOW_C_INT")

    # Wild cluster bootstrap p-values (B=999, valid for 14 clusters)
    print("Running wild cluster bootstrap (B=999) — this may take a few minutes...")
    wcb_epc = wild_cluster_bootstrap_pvalues(df_panel, "CURRENT_ENERGY_EFFICIENCY")
    wcb_co2 = wild_cluster_bootstrap_pvalues(df_panel, "CO2_EMISSIONS_CURRENT")
    wcb_belowc = wild_cluster_bootstrap_pvalues(df_panel, "BELOW_C_INT")

    epc_eff = extract_policy_effects(res_epc, wcb_pvalues=wcb_epc)
    co2_eff = extract_policy_effects(res_co2, wcb_pvalues=wcb_co2)
    belowc_eff = extract_policy_effects(res_belowc, wcb_pvalues=wcb_belowc)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    epc_eff.to_csv(TABLE_DIR / "policy_effects_epc.csv", index=False)
    co2_eff.to_csv(TABLE_DIR / "policy_effects_co2.csv", index=False)
    belowc_eff.to_csv(TABLE_DIR / "policy_effects_below_c.csv", index=False)
    register_artifact(REGISTRY_PATH, "table", TABLE_DIR / "policy_effects_epc.csv", {"model": "FE", "outcome": "EPC"})
    register_artifact(REGISTRY_PATH, "table", TABLE_DIR / "policy_effects_co2.csv", {"model": "FE", "outcome": "CO2"})
    register_artifact(REGISTRY_PATH, "table", TABLE_DIR / "policy_effects_below_c.csv", {"model": "FE", "outcome": "BelowC"})

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

    # Event study: year-by-year coefficients (2008 = reference)
    print("Fitting event study models...")
    es_epc = fit_event_study(df_panel, "CURRENT_ENERGY_EFFICIENCY")
    es_co2 = fit_event_study(df_panel, "CO2_EMISSIONS_CURRENT")
    es_belowc = fit_event_study(df_panel, "BELOW_C_INT")

    es_epc_tbl = extract_event_study(es_epc)
    es_co2_tbl = extract_event_study(es_co2)
    es_belowc_tbl = extract_event_study(es_belowc)

    es_epc_tbl.to_csv(TABLE_DIR / "event_study_epc.csv", index=False)
    es_co2_tbl.to_csv(TABLE_DIR / "event_study_co2.csv", index=False)
    es_belowc_tbl.to_csv(TABLE_DIR / "event_study_belowc.csv", index=False)
    register_artifact(REGISTRY_PATH, "table", TABLE_DIR / "event_study_epc.csv", {"model": "ES", "outcome": "EPC"})
    register_artifact(REGISTRY_PATH, "table", TABLE_DIR / "event_study_co2.csv", {"model": "ES", "outcome": "CO2"})
    register_artifact(REGISTRY_PATH, "table", TABLE_DIR / "event_study_belowc.csv", {"model": "ES", "outcome": "BelowC"})

    plot_event_study(
        es_epc_tbl,
        title="Event study: EPC efficiency score by year (2008 = reference)",
        y_label="Change in EPC score",
        out_path=FIG_DIR / "figure_6_5_event_study_epc.png",
    )
    plot_event_study(
        es_co2_tbl,
        title="Event study: CO₂ emissions by year (2008 = reference)",
        y_label="Change in CO₂ emissions (tonnes/year)",
        out_path=FIG_DIR / "figure_6_6_event_study_co2.png",
    )
    plot_event_study(
        es_belowc_tbl,
        title="Event study: Probability below EPC C by year (2008 = reference)",
        y_label="Change in probability",
        out_path=FIG_DIR / "figure_6_7_event_study_belowc.png",
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
