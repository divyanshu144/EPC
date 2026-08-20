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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

from config import REGISTRY_PATH, ROOT
from ew_housing_energy_impact.registry import register_artifact
from scripts.pipeline.policy import POLICY_ORDER

DATA_PATH = (
    ROOT / "data" / "processed" / "ew_epc_core_clean_sw" / "ew_epc_core_clean_sw_imputed.csv"
)
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

# Raw columns referenced by BASE_RHS. Used by the wild cluster bootstrap to
# reconstruct, positionally, the rows PanelOLS keeps (it drops any row with NaN
# in a model term). Keep in sync with _CONTROLS / BASE_RHS above.
_BASE_RHS_COLS: list[str] = [
    "POLICY_PERIOD",
    "PROPERTY_TYPE",
    "AGE_BAND_STD",
    "TENURE_STD",
    "MAIN_FUEL_STD",
    "LOG_FLOOR_AREA",
    "NEW_BUILD_SHARE",
]

# Event-study specification: year dummies (2008 = reference, dropped by patsy).
# Produces one coefficient per year, showing exactly when trends shift.
EVENT_STUDY_RHS = "C(YEAR_TERM)" + _CONTROLS

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
    df["POLICY_PERIOD"] = pd.Categorical(df["POLICY_PERIOD"], categories=POLICY_ORDER, ordered=True)

    # EVENT_STUDY_RHS refers to C(YEAR), but set_index moves YEAR out of the
    # columns and formulaic resolves formula terms against columns only — so the
    # event study raised "NameError: name 'YEAR' is not defined". Keep an
    # explicit column alongside the index level. Named separately rather than
    # using drop=False so the index level and the regressor are unambiguous.
    df["YEAR_TERM"] = df["YEAR"]

    # Drop rows outside the defined policy periods, explicitly and loudly.
    #
    # policy_period() returns "Other" beyond the scheme's final year, which the
    # Categorical above turns into NaN. Those rows were being discarded anyway —
    # PanelOLS drops any row with NaN in a model term — but silently, and
    # MixedLM in clustering_from_random_effects fails outright on them with an
    # opaque "Shape mismatch between endog/exog and extra 2d arrays".
    #
    # Excluding here makes the scope of the analysis a stated decision rather
    # than a side effect of NaN handling. The excluded year is also partial, so
    # including it would understate that year in the event study regardless.
    outside = df["POLICY_PERIOD"].isna()
    if outside.any():
        years = sorted(df.loc[outside, "YEAR"].dropna().unique())
        print(
            f"Excluding {int(outside.sum()):,} rows outside the defined policy "
            f"periods (years: {', '.join(str(y) for y in years)})."
        )
        df = df[~outside]

    df = df.set_index(["LOCAL_AUTHORITY", "YEAR"]).sort_index()
    return df


def fit_fe(df_panel: pd.DataFrame, dep_var: str):
    formula = f"{dep_var} ~ 1 + {BASE_RHS}"
    clusters = pd.Series(
        df_panel.index.get_level_values("LOCAL_AUTHORITY"),
        index=df_panel.index,
        name="LOCAL_AUTHORITY",
    )
    model = PanelOLS.from_formula(formula, data=df_panel, drop_absorbed=True, check_rank=False)
    return model.fit(cov_type="clustered", clusters=clusters)


def wild_cluster_bootstrap_pvalues(
    df_panel: pd.DataFrame,
    dep_var: str,
    B: int = 999,
    seed: int = 42,
) -> pd.Series:
    """Wild cluster bootstrap p-values (Rademacher weights).

    Valid for small numbers of clusters (Cameron, Gelbach & Miller 2008;
    MacKinnon & Webb 2017). Returns a Series of p-values indexed by parameter
    name, for terms containing 'POLICY_PERIOD'.

    Method
    ------
    1. Fit the FE model once to obtain fitted values, residuals and the
       full-sample coefficient estimates.
    2. In each bootstrap iteration draw one Rademacher weight (+1/-1) *per
       cluster* (same sign for every observation in an LA), build the perturbed
       outcome ``y* = fitted + weight * residual``, and refit.
    3. Recentre the bootstrap t-statistic on the full-sample estimate,
       ``t* = (beta*_b - beta_hat) / se*_b``. Recentring is essential: without
       it the bootstrap statistics cluster around the observed estimate rather
       than the null, and the resulting p-values are biased toward ~0.5. (This
       is the WCB-U / restricted-recentred variant; it imposes the null on the
       statistic, not on the resampling DGP.)
    4. The p-value uses the finite-sample correction
       ``(1 + #{|t*| >= |t_obs|}) / (n_boot + 1)`` so it can never be exactly 0.

    Robustness: linearmodels returns ``fitted_values`` as an (n x 1) DataFrame
    and ``resids`` as a Series, and PanelOLS drops rows with NaN in any RHS term
    (e.g. NEW_BUILD_SHARE / YEAR). Both are therefore aligned to the *estimation*
    index, which can be shorter than ``df_panel``; we convert to 1-D arrays and
    align cluster weights to that estimation index to avoid shape errors.
    """
    rng = np.random.default_rng(seed)

    # Fit baseline model.
    res = fit_fe(df_panel, dep_var)

    # 1-D arrays aligned to the estimation sample (handles DataFrame fitted
    # values and any NaN-dropped rows).
    fitted_arr = res.fitted_values.iloc[:, 0].to_numpy()
    resid_arr = res.resids.to_numpy()

    # Reconstruct, positionally, the rows PanelOLS used. The panel index has
    # duplicate (LA, YEAR) pairs, so label-based assignment is impossible; we
    # work with a boolean row mask instead. PanelOLS drops rows with NaN in any
    # model term, so the kept rows are exactly the non-NaN rows of _BASE_RHS_COLS
    # + dep_var.
    est_cols = [c for c in _BASE_RHS_COLS + [dep_var] if c in df_panel.columns]
    est_mask = df_panel[est_cols].notna().all(axis=1).to_numpy()
    if int(est_mask.sum()) != resid_arr.shape[0]:
        raise RuntimeError(
            f"WCB could not reconstruct the estimation sample for {dep_var}: "
            f"{int(est_mask.sum())} non-NaN rows vs {resid_arr.shape[0]} residuals. "
            "Check that _BASE_RHS_COLS matches BASE_RHS."
        )

    # Cluster index for the kept rows, factorised once for fast weight gather.
    las_used = df_panel.index.get_level_values("LOCAL_AUTHORITY").to_numpy()[est_mask]
    cluster_codes, unique_las = pd.factorize(las_used)
    n_clusters = len(unique_las)

    # Observed t-statistics (H0: beta = 0) for policy-period terms.
    policy_terms = [p for p in res.params.index if "POLICY_PERIOD" in p]
    beta_hat = res.params
    t_obs = (res.params[policy_terms] / res.std_errors[policy_terms]).abs()

    t_boot = {term: [] for term in policy_terms}

    # Reuse a single working frame; only the dependent column is overwritten
    # each iteration (avoids copying the full ~1.8M-row panel B times).
    df_boot = df_panel.copy()
    dep_full = df_panel[dep_var].to_numpy(dtype=float)

    for _ in range(B):
        try:
            # One Rademacher weight per cluster, broadcast to its observations.
            weights = rng.choice([-1.0, 1.0], size=n_clusters)
            obs_weights = weights[cluster_codes]

            # Perturbed outcome: overwrite kept rows positionally; dropped rows
            # keep their (NaN) value and are dropped again on refit.
            dep_boot = dep_full.copy()
            dep_boot[est_mask] = fitted_arr + obs_weights * resid_arr
            df_boot[dep_var] = dep_boot

            res_b = fit_fe(df_boot, dep_var)
            for term in policy_terms:
                if term in res_b.params.index:
                    t_star = (res_b.params[term] - beta_hat[term]) / res_b.std_errors[term]
                    t_boot[term].append(abs(t_star))
        except Exception:
            # Skip a degenerate refit (e.g. rank deficiency under perturbation);
            # the p-value below uses the count of successful replications.
            continue

    pvals = {}
    for term in policy_terms:
        boot_arr = np.asarray(t_boot[term], dtype=float)
        if boot_arr.size == 0:
            pvals[term] = np.nan
        else:
            pvals[term] = (1 + np.sum(boot_arr >= t_obs[term])) / (boot_arr.size + 1)

    return pd.Series(pvals, name="wcb_pvalue")


def extract_policy_effects(res, wcb_pvalues: pd.Series | None = None, prefix="C(POLICY_PERIOD)"):
    """Extract policy-period coefficients, clustered SEs, 95% CIs, and WCB p-values."""
    params = res.params.filter(like=prefix)
    se = res.std_errors.filter(like=prefix)
    clustered_pvals = res.pvalues.filter(like=prefix)
    out = pd.concat([params, se, clustered_pvals], axis=1).reset_index()
    out.columns = ["term", "coef", "se", "clustered_pval"]
    # lower_95/upper_95 are asymptotic clustered intervals (±1.96·SE) reported
    # for comparability and visualisation. With 14 clusters they are unreliable;
    # the wcb_pval column (wild cluster bootstrap) is the primary inference.
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
    model = PanelOLS.from_formula(formula, data=df_panel, drop_absorbed=True, check_rank=False)
    return model.fit(cov_type="clustered", clusters=clusters)


def extract_event_study(res, ref_year: int) -> pd.DataFrame:
    """Extract year coefficients and 95% CIs from an event-study result.

    Prepends the reference (omitted) year with coefficient = 0 so the plot
    anchors at the true baseline. ``ref_year`` must be the year patsy actually
    dropped — i.e. the minimum YEAR in the estimation panel — not a hardcoded
    value, otherwise every plotted coefficient is shifted relative to a baseline
    that may not exist in the data.

    Note: lower_95/upper_95 are asymptotic clustered intervals for visual
    reference only; formal inference uses the wild cluster bootstrap.

    Returns a DataFrame with columns: year, coef, se, lower_95, upper_95.
    """
    prefix = "C(YEAR_TERM)"
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

    # Prepend the actual reference year at zero (skip if it somehow appears as a
    # fitted term, to avoid double-counting).
    if ref_year not in set(out["year"]):
        ref_row = pd.DataFrame(
            [{"year": ref_year, "coef": 0.0, "se": 0.0, "lower_95": 0.0, "upper_95": 0.0}]
        )
        out = pd.concat([ref_row, out], ignore_index=True)
    out = out.sort_values("year").reset_index(drop=True)
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

    # CI ribbon. These are asymptotic clustered intervals (±1.96·SE) shown for
    # visual reference only — with 14 clusters they are unreliable, so formal
    # inference relies on the wild cluster bootstrap p-values, not this band.
    ax.fill_between(
        years,
        lo,
        hi,
        alpha=0.18,
        color="#2C7BB6",
        label="95% CI (asymptotic, clustered)",
    )

    # Point estimates connected by line
    ax.plot(years, coefs, color="#2C7BB6", linewidth=1.4, zorder=3)
    ax.scatter(years, coefs, color="#2C7BB6", s=40, zorder=4)

    # Baseline
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    # Policy-period boundary lines + period labels
    for break_year, _label in _PERIOD_BREAKS:
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
    ax.legend(loc="best", frameon=False, fontsize=8)

    fig.text(
        0.01,
        0.005,
        "CI band: asymptotic clustered ±1.96·SE (visual reference only). "
        "Formal inference uses wild cluster bootstrap p-values.",
        fontsize=7,
        color="#777777",
    )

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
    fig = plt.gcf()
    fig.text(
        0.01,
        0.005,
        "Error bars: asymptotic clustered ±1.96·SE (visual reference only). "
        "Formal inference uses wild cluster bootstrap p-values (wcb_pval).",
        fontsize=7,
        color="#777777",
    )
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


def clustering_from_random_effects(
    df_panel: pd.DataFrame, out_prefix: str, sample_n: int = 200_000
):
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


def main(bootstrap: int = 999) -> None:
    """Fit the policy models and write tables and figures.

    Args:
        bootstrap: wild cluster bootstrap replications. Set to 0 to skip the
            bootstrap and leave ``wcb_pval`` empty, reporting clustered SEs
            only.

            Skipping is a deliberate mode, not a shortcut. On the full
            2.68M-row panel one PanelOLS fit takes ~19s, so B=999 across three
            dependent variables is ~15.7 hours, while every other output in
            this module takes seconds. Separating them lets the coefficients,
            event study, heterogeneity splits and clustering land immediately
            and the bootstrap run afterwards.

            Anything written with bootstrap=0 must not be presented as final
            inference: with 14 clusters the clustered SEs are precisely what
            the bootstrap exists to replace (Cameron, Gelbach & Miller 2008).
    """
    df = load_data()
    df_panel = prepare_panel(df)

    # Main FE models
    print("Fitting FE models...")
    res_epc = fit_fe(df_panel, "CURRENT_ENERGY_EFFICIENCY")
    res_co2 = fit_fe(df_panel, "CO2_EMISSIONS_CURRENT")
    res_belowc = fit_fe(df_panel, "BELOW_C_INT")

    if bootstrap:
        print(f"Running wild cluster bootstrap (B={bootstrap})...")
        wcb_epc = wild_cluster_bootstrap_pvalues(df_panel, "CURRENT_ENERGY_EFFICIENCY", B=bootstrap)
        wcb_co2 = wild_cluster_bootstrap_pvalues(df_panel, "CO2_EMISSIONS_CURRENT", B=bootstrap)
        wcb_belowc = wild_cluster_bootstrap_pvalues(df_panel, "BELOW_C_INT", B=bootstrap)
    else:
        print("Skipping wild cluster bootstrap (bootstrap=0); wcb_pval will be empty.")
        print("  Clustered SEs with 14 clusters are NOT reliable inference on their own.")
        wcb_epc = wcb_co2 = wcb_belowc = None

    epc_eff = extract_policy_effects(res_epc, wcb_pvalues=wcb_epc)
    co2_eff = extract_policy_effects(res_co2, wcb_pvalues=wcb_co2)
    belowc_eff = extract_policy_effects(res_belowc, wcb_pvalues=wcb_belowc)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    epc_eff.to_csv(TABLE_DIR / "policy_effects_epc.csv", index=False)
    co2_eff.to_csv(TABLE_DIR / "policy_effects_co2.csv", index=False)
    belowc_eff.to_csv(TABLE_DIR / "policy_effects_below_c.csv", index=False)
    register_artifact(
        REGISTRY_PATH,
        "table",
        TABLE_DIR / "policy_effects_epc.csv",
        {"model": "FE", "outcome": "EPC"},
    )
    register_artifact(
        REGISTRY_PATH,
        "table",
        TABLE_DIR / "policy_effects_co2.csv",
        {"model": "FE", "outcome": "CO2"},
    )
    register_artifact(
        REGISTRY_PATH,
        "table",
        TABLE_DIR / "policy_effects_below_c.csv",
        {"model": "FE", "outcome": "BelowC"},
    )

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

    # Event study: year-by-year coefficients. The reference is the omitted first
    # level of C(YEAR), i.e. the minimum YEAR actually present in the panel.
    ref_year = int(df_panel.index.get_level_values("YEAR").min())
    print(f"Fitting event study models (reference year = {ref_year})...")
    es_epc = fit_event_study(df_panel, "CURRENT_ENERGY_EFFICIENCY")
    es_co2 = fit_event_study(df_panel, "CO2_EMISSIONS_CURRENT")
    es_belowc = fit_event_study(df_panel, "BELOW_C_INT")

    es_epc_tbl = extract_event_study(es_epc, ref_year)
    es_co2_tbl = extract_event_study(es_co2, ref_year)
    es_belowc_tbl = extract_event_study(es_belowc, ref_year)

    es_epc_tbl.to_csv(TABLE_DIR / "event_study_epc.csv", index=False)
    es_co2_tbl.to_csv(TABLE_DIR / "event_study_co2.csv", index=False)
    es_belowc_tbl.to_csv(TABLE_DIR / "event_study_belowc.csv", index=False)
    register_artifact(
        REGISTRY_PATH, "table", TABLE_DIR / "event_study_epc.csv", {"model": "ES", "outcome": "EPC"}
    )
    register_artifact(
        REGISTRY_PATH, "table", TABLE_DIR / "event_study_co2.csv", {"model": "ES", "outcome": "CO2"}
    )
    register_artifact(
        REGISTRY_PATH,
        "table",
        TABLE_DIR / "event_study_belowc.csv",
        {"model": "ES", "outcome": "BelowC"},
    )

    plot_event_study(
        es_epc_tbl,
        title=f"Event study: EPC efficiency score by year ({ref_year} = reference)",
        y_label="Change in EPC score",
        out_path=FIG_DIR / "figure_6_5_event_study_epc.png",
    )
    plot_event_study(
        es_co2_tbl,
        title=f"Event study: CO₂ emissions by year ({ref_year} = reference)",
        y_label="Change in CO₂ emissions (tonnes/year)",
        out_path=FIG_DIR / "figure_6_6_event_study_co2.png",
    )
    plot_event_study(
        es_belowc_tbl,
        title=f"Event study: Probability below EPC C by year ({ref_year} = reference)",
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
