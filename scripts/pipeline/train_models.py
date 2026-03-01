#!/usr/bin/env python3
"""
Predictive modelling for EPC efficiency and EPC-below-C classification.
Aligned with the dissertation report (GBR/MLP/XGBoost, feature importance, SHAP).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor

import shap
from xgboost import XGBClassifier
import joblib

from config import ROOT, REGISTRY_PATH
from ew_housing_energy_impact.registry import register_artifact


DATA_PATH = ROOT / "data" / "processed" / "ew_epc_core_clean_sw" / "ew_epc_core_clean_sw_imputed.csv"
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"
ARTIFACT_DIR = ROOT / "reports" / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
MODEL_CARD_DIR = ROOT / "reports" / "model_cards"

RANDOM_STATE = 42
TARGET_N = 200_000

sns.set(style="whitegrid")


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if "BELOW_C_INT" not in df.columns:
        df["BELOW_C_INT"] = df["BELOW_C"].astype(int)
    if "LOG_FLOOR_AREA" not in df.columns:
        df["LOG_FLOOR_AREA"] = np.log(df["TOTAL_FLOOR_AREA"])
    return df


def sample_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by BUILDING_REFERENCE_NUMBER to prevent leakage across train/test.

    The same physical dwelling can have multiple EPC certificates across years.
    A row-level random split would allow the same property to appear in both
    train and test sets, inflating evaluation metrics. Splitting on the property
    identifier ensures the test set contains only unseen properties.

    Returns (train_df, test_df) where test is ~20% of unique properties.
    """
    if "BUILDING_REFERENCE_NUMBER" not in df.columns:
        # Fallback for datasets without the ID column: row-level split with warning.
        import warnings
        warnings.warn(
            "BUILDING_REFERENCE_NUMBER not found — falling back to row-level split. "
            "Test metrics may be optimistic if properties repeat across years.",
            UserWarning,
            stacklevel=2,
        )
        train_df, test_df = train_test_split(
            df, test_size=0.2, stratify=df["BELOW_C_INT"], random_state=RANDOM_STATE
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    # Property-level split: 80% of unique properties → train, 20% → test.
    unique_props = df["BUILDING_REFERENCE_NUMBER"].dropna().unique()
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(unique_props)
    n_test = max(1, int(len(unique_props) * 0.2))
    test_props = set(unique_props[:n_test])

    test_mask = df["BUILDING_REFERENCE_NUMBER"].isin(test_props)
    train_df = df[~test_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    # Down-sample train to TARGET_N if needed (preserve BELOW_C balance).
    if len(train_df) > TARGET_N:
        train_df, _ = train_test_split(
            train_df,
            train_size=TARGET_N,
            stratify=train_df["BELOW_C_INT"],
            random_state=RANDOM_STATE,
        )
        train_df = train_df.reset_index(drop=True)

    return train_df, test_df


def build_preprocessor(cat_features: list[str], num_features: list[str]) -> ColumnTransformer:
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numeric_transformer = StandardScaler()
    return ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, cat_features),
            ("num", numeric_transformer, num_features),
        ]
    )


def evaluate_regression(name, model, X_train, y_train, X_test, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    return {
        "model": name,
        "rmse_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "rmse_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "mae_test": mean_absolute_error(y_test, y_pred_test),
        "r2_train": r2_score(y_train, y_pred_train),
        "r2_test": r2_score(y_test, y_pred_test),
    }


def evaluate_classifier(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def main() -> None:
    train_df, test_df = sample_data(load_data())

    feature_cols = [
        "PROPERTY_TYPE",
        "AGE_BAND_STD",
        "TENURE_STD",
        "MAIN_FUEL_STD",
        "LOG_FLOOR_AREA",
    ]
    target_reg = "CURRENT_ENERGY_EFFICIENCY"
    target_clf = "BELOW_C_INT"

    train_df = train_df.dropna(subset=feature_cols + [target_reg, target_clf])
    test_df = test_df.dropna(subset=feature_cols + [target_reg, target_clf])

    X_train = train_df[feature_cols]
    y_reg_train = train_df[target_reg]
    y_clf_train = train_df[target_clf]

    X_test = test_df[feature_cols]
    y_reg_test = test_df[target_reg]
    y_clf_test = test_df[target_clf]

    cat_features = ["PROPERTY_TYPE", "AGE_BAND_STD", "TENURE_STD", "MAIN_FUEL_STD"]
    num_features = ["LOG_FLOOR_AREA"]
    preprocessor = build_preprocessor(cat_features, num_features)

    # --- Regression models ---
    reg_models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=5000),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=300,
            random_state=RANDOM_STATE,
            early_stopping=True,
            n_iter_no_change=10,
        ),
    }

    reg_results = []
    for name, base_model in reg_models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", base_model)])
        pipe.fit(X_train, y_reg_train)
        reg_results.append(
            evaluate_regression(name, pipe, X_train, y_reg_train, X_test, y_reg_test)
        )

    # Tuned GBR
    gbr_pipe = Pipeline(
        steps=[("preprocess", preprocessor), ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))]
    )
    gbr_param_grid = {
        "model__n_estimators": [150, 250, 350],
        "model__max_depth": [2, 3],
        "model__learning_rate": [0.05, 0.1],
    }
    gbr_search = GridSearchCV(
        gbr_pipe, gbr_param_grid, scoring="neg_root_mean_squared_error", cv=3, n_jobs=-1
    )
    gbr_search.fit(X_train, y_reg_train)
    best_gbr = gbr_search.best_estimator_
    reg_results.append(
        evaluate_regression("GBR (tuned)", best_gbr, X_train, y_reg_train, X_test, y_reg_test)
    )

    # Tuned MLP
    mlp_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", MLPRegressor(random_state=RANDOM_STATE, early_stopping=True))])
    mlp_param = {
        "model__hidden_layer_sizes": [(128, 64), (64, 32), (128, 32)],
        "model__alpha": [0.0001, 0.001],
        "model__max_iter": [300, 400],
    }
    mlp_search = RandomizedSearchCV(
        mlp_pipe,
        mlp_param,
        n_iter=6,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    mlp_search.fit(X_train, y_reg_train)
    best_mlp = mlp_search.best_estimator_
    reg_results.append(
        evaluate_regression("MLP (tuned)", best_mlp, X_train, y_reg_train, X_test, y_reg_test)
    )

    reg_results_df = pd.DataFrame(reg_results).sort_values("rmse_test").reset_index(drop=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    reg_results_df.to_csv(TABLE_DIR / "regression_model_performance.csv", index=False)
    register_artifact(REGISTRY_PATH, "table", TABLE_DIR / "regression_model_performance.csv", {"task": "regression"})

    # Feature importance (GBR tuned)
    preprocessor = best_gbr.named_steps["preprocess"]
    gb_model = best_gbr.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importances = gb_model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    fi_df.to_csv(TABLE_DIR / "gbr_feature_importance.csv", index=False)
    register_artifact(REGISTRY_PATH, "table", TABLE_DIR / "gbr_feature_importance.csv", {"model": "GBR"})

    # Grouped importances
    def aggregate_importances(fi):
        rows = []
        for feat, imp in zip(fi["feature"], fi["importance"]):
            if feat.startswith("cat__"):
                base = feat.split("__")[1].split("_")[0]
            elif feat.startswith("num__"):
                base = feat.split("__")[1]
            else:
                base = feat
            rows.append({"group": base, "importance": imp})
        return pd.DataFrame(rows).groupby("group", as_index=False)["importance"].sum()

    fi_grouped = aggregate_importances(fi_df).sort_values("importance", ascending=False)
    fi_grouped.to_csv(TABLE_DIR / "gbr_feature_importance_grouped.csv", index=False)
    register_artifact(REGISTRY_PATH, "table", TABLE_DIR / "gbr_feature_importance_grouped.csv", {"model": "GBR"})

    plt.figure(figsize=(8, 5))
    top = fi_grouped.head(10).sort_values("importance")
    plt.barh(top["group"], top["importance"])
    plt.xlabel("Total feature importance")
    plt.title("Gradient Boosting – feature importance by variable group")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "figure_5_1_gbr_feature_importance.png", dpi=200)
    plt.close()
    register_artifact(REGISTRY_PATH, "figure", FIG_DIR / "figure_5_1_gbr_feature_importance.png", {"model": "GBR"})

    # SHAP for GBR
    X_train_trans = best_gbr.named_steps["preprocess"].transform(X_train)
    explainer = shap.TreeExplainer(gb_model)
    shap_values = explainer.shap_values(X_train_trans)
    shap.summary_plot(
        shap_values,
        X_train_trans,
        feature_names=preprocessor.get_feature_names_out(),
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_5_2_shap_summary.png", dpi=200)
    plt.close()
    register_artifact(REGISTRY_PATH, "figure", FIG_DIR / "figure_5_2_shap_summary.png", {"model": "GBR", "type": "shap_summary"})

    # --- Classification models ---
    clf_models = {
        "LogisticRegression": LogisticRegression(max_iter=500, n_jobs=-1),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }

    clf_results = {}
    clf_metrics = []
    for name, base_model in clf_models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", base_model)])
        pipe.fit(X_train, y_clf_train)
        clf_results[name] = pipe
        clf_metrics.append(evaluate_classifier(name, pipe, X_test, y_clf_test))

    clf_results_df = pd.DataFrame(clf_metrics).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    clf_results_df.to_csv(TABLE_DIR / "classification_model_performance.csv", index=False)
    register_artifact(REGISTRY_PATH, "table", TABLE_DIR / "classification_model_performance.csv", {"task": "classification"})

    best_name = clf_results_df.loc[0, "model"]
    best_pipe = clf_results[best_name]

    # Confusion matrix and ROC
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_estimator(best_pipe, X_test, y_clf_test, ax=axes[0], cmap="Blues")
    RocCurveDisplay.from_estimator(best_pipe, X_test, y_clf_test, ax=axes[1])
    axes[0].set_title("Confusion Matrix")
    axes[1].set_title("ROC Curve")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure_5_3_xgb_confusion_roc.png", dpi=200)
    plt.close()
    register_artifact(REGISTRY_PATH, "figure", FIG_DIR / "figure_5_3_xgb_confusion_roc.png", {"model": best_name, "type": "confusion_roc"})

    # XGBoost feature importance
    if best_name == "XGBoost":
        xgb_model = best_pipe.named_steps["model"]
        ohe = best_pipe.named_steps["preprocess"].named_transformers_["cat"]
        feature_names_clf = list(ohe.get_feature_names_out(cat_features)) + num_features
        imp = pd.Series(xgb_model.feature_importances_, index=feature_names_clf)
        imp.sort_values(ascending=False).to_csv(TABLE_DIR / "xgb_feature_importance.csv")
        register_artifact(REGISTRY_PATH, "table", TABLE_DIR / "xgb_feature_importance.csv", {"model": "XGBoost"})

    # Save models
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_gbr_path = MODEL_DIR / "gbr_tuned.joblib"
    model_mlp_path = MODEL_DIR / "mlp_tuned.joblib"
    model_xgb_path = MODEL_DIR / "xgb_best.joblib"
    joblib.dump(best_gbr, model_gbr_path)
    joblib.dump(best_mlp, model_mlp_path)
    joblib.dump(best_pipe, model_xgb_path)
    register_artifact(REGISTRY_PATH, "model", model_gbr_path, {"model": "GBR (tuned)"})
    register_artifact(REGISTRY_PATH, "model", model_mlp_path, {"model": "MLP (tuned)"})
    register_artifact(REGISTRY_PATH, "model", model_xgb_path, {"model": best_name})

    # Model cards (markdown)
    MODEL_CARD_DIR.mkdir(parents=True, exist_ok=True)
    def write_model_card(name: str, metrics: dict, fig_refs: list[str], table_refs: list[str]) -> None:
        card_path = MODEL_CARD_DIR / f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}.md"
        lines = [
            f"# Model Card: {name}",
            "",
            "## Summary",
            "Predictive model aligned with the dissertation report.",
            "",
            "## Metrics",
        ]
        for k, v in metrics.items():
            if k == "model":
                continue
            lines.append(f"- `{k}`: {v:.4f}" if isinstance(v, (int, float)) else f"- `{k}`: {v}")
        lines += ["", "## Figures"]
        for f in fig_refs:
            lines.append(f"- `{f}`")
        lines += ["", "## Tables"]
        for t in table_refs:
            lines.append(f"- `{t}`")
        card_path.write_text("\n".join(lines))
        register_artifact(REGISTRY_PATH, "model_card", card_path, {"model": name})

    # Model cards (best reg + GBR + best clf)
    best_reg = reg_results_df.iloc[0].to_dict()
    write_model_card(
        best_reg["model"],
        best_reg,
        ["reports/figures/figure_5_1_gbr_feature_importance.png", "reports/figures/figure_5_2_shap_summary.png"],
        ["reports/tables/regression_model_performance.csv", "reports/tables/gbr_feature_importance_grouped.csv"],
    )

    # Explicit GBR card (even if not best)
    gbr_row = reg_results_df[reg_results_df["model"].str.contains("GBR")].head(1)
    if not gbr_row.empty:
        gbr_metrics = gbr_row.iloc[0].to_dict()
        write_model_card(
            "GBR (tuned)",
            gbr_metrics,
            ["reports/figures/figure_5_1_gbr_feature_importance.png", "reports/figures/figure_5_2_shap_summary.png"],
            ["reports/tables/regression_model_performance.csv", "reports/tables/gbr_feature_importance_grouped.csv"],
        )

    best_clf = clf_results_df.iloc[0].to_dict()
    write_model_card(
        best_clf["model"],
        best_clf,
        ["reports/figures/figure_5_3_xgb_confusion_roc.png"],
        ["reports/tables/classification_model_performance.csv"],
    )

    print("✓ Predictive modelling completed.")


if __name__ == "__main__":
    main()
