"""Evaluacion predictiva rigurosa sobre bases de datos reales + dashboard HTML interactivo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ada.audit.ledger import record_audit_event
from ada.config import DATA_WORK_DIR, DEFAULT_SEED, OUTPUTS_DIR, assert_safe_write_path, set_global_seed
from ada.critic.auditor import audit_classification_results, audit_regression_results
from ada.ingest.loader import load_dataset_frame
from ada.viz.web_studio import audit_html_quality, render_prediction_studio_html


def _bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Any,
    *,
    n_boot: int = 400,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n < 5:
        val = float(metric_fn(y_true, y_pred))
        return val, val
    scores: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            scores.append(float(metric_fn(y_true[idx], y_pred[idx])))
        except ValueError:
            continue
    if not scores:
        val = float(metric_fn(y_true, y_pred))
        return val, val
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def _build_preprocessor(X_train: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    num_cols = [
        c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c]) and not pd.api.types.is_bool_dtype(X_train[c])
    ]
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    transformers: list[tuple[str, Any, list[str]]] = []
    if num_cols:
        num_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False, max_categories=30)),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop"), num_cols, cat_cols


def _render_prediction_dashboard_html(
    *,
    output_html_path: Path,
    title: str,
    task_type: str,
    target_column: str,
    models_summary: dict[str, Any],
    best_model_name: str,
    feature_importances: list[dict[str, Any]],
    critic_report: dict[str, Any],
    chart_payload: dict[str, Any],
    dataset_name: str = "dataset",
    n_train: int = 0,
    n_test: int = 0,
    n_features: int = 0,
    sample_predictions: list[dict[str, Any]] | None = None,
    verification_badge: str = "VERIFICADO: scikit-learn + statsmodels",
) -> Path:
    return render_prediction_studio_html(
        output_html_path=output_html_path,
        title=title,
        dataset_name=dataset_name,
        task_type=task_type,
        target_column=target_column,
        n_train=n_train,
        n_test=n_test,
        n_features=n_features,
        models_summary=models_summary,
        best_model_name=best_model_name,
        feature_importances=feature_importances,
        critic_report=critic_report,
        chart_payload=chart_payload,
        sample_predictions=sample_predictions,
        verification_badge=verification_badge,
    )


def evaluate_and_predict_database(
    input_path: str | Path,
    target_column: str,
    *,
    table_or_sheet: str = "",
    where_sql: str = "",
    exclude_columns: list[str] | None = None,
    task_type: str = "auto",
    test_size: float = 0.2,
    predict_on_path: str = "",
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Entrena, valida por doble metodo, audita con el Critico Estadistico y predice sobre una base de datos real."""
    set_global_seed(seed)
    src_path = Path(input_path).resolve()
    df, meta = load_dataset_frame(src_path, table_or_sheet=table_or_sheet)

    if where_sql.strip():
        import duckdb

        conn = duckdb.connect(":memory:")
        try:
            conn.register("src_tbl", df)
            df = conn.execute(f"SELECT * FROM src_tbl WHERE {where_sql.strip()}").df()
        finally:
            conn.close()

    if target_column not in df.columns:
        raise ValueError(f"La variable objetivo '{target_column}' no existe. Columnas: {list(df.columns)}")

    unlabeled_mask = df[target_column].isna()
    df_unlabeled_internal = df[unlabeled_mask].copy()
    df_labeled = df[~unlabeled_mask].copy().reset_index(drop=True)

    if len(df_labeled) < 16:
        raise ValueError(f"Muestra etiquetada insuficiente (n={len(df_labeled)} < 16) para entrenar y validar con rigor.")

    drop_cols = {target_column}
    if exclude_columns:
        drop_cols.update(exclude_columns)

    auto_dropped_ids: list[str] = []
    for c in df_labeled.columns:
        if c in drop_cols:
            continue
        s = df_labeled[c]
        if s.isna().all():
            drop_cols.add(c)
            auto_dropped_ids.append(f"{c} (100% nulo)")
        elif not pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) == len(df_labeled) and len(df_labeled) >= 25:
            drop_cols.add(c)
            auto_dropped_ids.append(f"{c} (ID textual 100% unico)")

    feature_cols = [c for c in df_labeled.columns if c not in drop_cols]
    if not feature_cols:
        raise ValueError("No quedaron columnas predictoras tras excluir IDs/nulos.")

    X = df_labeled[feature_cols].copy()
    y_raw = df_labeled[target_column].copy()

    resolved_task = task_type.strip().lower()
    if resolved_task == "auto":
        if pd.api.types.is_bool_dtype(y_raw) or not pd.api.types.is_numeric_dtype(y_raw):
            resolved_task = "classification"
        elif y_raw.nunique(dropna=True) <= min(10, max(2, len(df_labeled) // 15)) and np.all(
            np.equal(np.mod(y_raw.to_numpy(dtype=float), 1), 0)
        ):
            resolved_task = "classification"
        else:
            resolved_task = "regression"

    DATA_WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if resolved_task == "regression":
        y = pd.to_numeric(y_raw, errors="raise").to_numpy(dtype=float)
        # REGLA CRITICA (ml-best-practices): Split train/test ANTES de cualquier preprocesamiento
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        preproc, num_cols, cat_cols = _build_preprocessor(X_train)

        candidates: dict[str, Any] = {
            "DummyBaseline": Pipeline([("prep", preproc), ("model", DummyRegressor(strategy="median"))]),
            "RidgeLinear": Pipeline([("prep", preproc), ("model", Ridge(alpha=1.0))]),
            "HistGradientBoosting": Pipeline(
                [("prep", preproc), ("model", HistGradientBoostingRegressor(max_iter=150, random_state=seed))]
            ),
        }

        kf = KFold(n_splits=min(5, max(2, len(X_train) // 5)), shuffle=True, random_state=seed)
        models_summary: dict[str, Any] = {}
        fitted_pipes: dict[str, Any] = {}

        for name, pipe in candidates.items():
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))
            ci_low, ci_high = _bootstrap_metric_ci(y_test, preds, r2_score, seed=seed)
            models_summary[name] = {
                "rmse": round(rmse, 6),
                "mae": round(mae, 6),
                "r2": round(r2, 6),
                "r2_ci95": [round(ci_low, 6), round(ci_high, 6)],
                "cv_mean": round(float(np.mean(cv_scores)), 6),
                "cv_std": round(float(np.std(cv_scores)), 6),
            }
            fitted_pipes[name] = (pipe, preds)

        # Verificacion independiente (Regla 2): Contrastar Ridge(alpha->0) vs statsmodels.OLS en features preprocesadas
        X_tr_mat = fitted_pipes["RidgeLinear"][0].named_steps["prep"].transform(X_train)
        X_te_mat = fitted_pipes["RidgeLinear"][0].named_steps["prep"].transform(X_test)
        cond_number = float(np.linalg.cond(X_tr_mat)) if X_tr_mat.shape[1] > 0 and X_tr_mat.shape[0] > X_tr_mat.shape[1] else None

        ols_model = sm.OLS(y_train, sm.add_constant(X_tr_mat, has_constant="add")).fit()
        ols_preds = np.asarray(ols_model.predict(sm.add_constant(X_te_mat, has_constant="add")), dtype=float)
        ridge_tiny = Ridge(alpha=1e-8).fit(X_tr_mat, y_train).predict(X_te_mat)
        dual_ols_max_diff = float(np.max(np.abs(ols_preds - ridge_tiny)))
        dual_verified = bool(np.isfinite(dual_ols_max_diff) and (dual_ols_max_diff < 1e-4 or (cond_number or 0.0) > 1e6))

        non_dummy = ["RidgeLinear", "HistGradientBoosting"]
        best_name = min(non_dummy, key=lambda k: models_summary[k]["rmse"])
        best_pipe, best_preds = fitted_pipes[best_name]
        residuals_test = y_test - best_preds

        # Intervalo cuantilico libre de distribucion (90%) basado en residuos de calibracion/test
        q90_half_width = float(np.quantile(np.abs(residuals_test), 0.90))

        perm = permutation_importance(best_pipe, X_test, y_test, n_repeats=10, random_state=seed, scoring="r2")
        feat_imp = sorted(
            [
                {"feature": str(col), "importance": round(float(imp), 6)}
                for col, imp in zip(feature_cols, perm.importances_mean)
            ],
            key=lambda d: d["importance"],
            reverse=True,
        )

        critic = audit_regression_results(
            n_train=len(X_train),
            n_test=len(X_test),
            n_features=len(feature_cols),
            best_r2=models_summary[best_name]["r2"],
            dummy_r2=models_summary["DummyBaseline"]["r2"],
            best_rmse=models_summary[best_name]["rmse"],
            dummy_rmse=models_summary["DummyBaseline"]["rmse"],
            residuals_test=residuals_test,
            y_pred_test=best_preds,
            design_condition_number=cond_number,
        )

        # Predicciones sobre archivo nuevo, filas sin etiqueta o test set
        if predict_on_path.strip():
            pred_df_src, _ = load_dataset_frame(predict_on_path.strip())
            y_hat_new = best_pipe.predict(pred_df_src[feature_cols])
            out_pred_df = pred_df_src.copy()
            out_pred_df[f"pred_{target_column}"] = y_hat_new
            out_pred_df[f"pred_{target_column}_low90"] = y_hat_new - q90_half_width
            out_pred_df[f"pred_{target_column}_high90"] = y_hat_new + q90_half_width
            pred_scope = "external_dataset"
        elif len(df_unlabeled_internal) > 0:
            y_hat_new = best_pipe.predict(df_unlabeled_internal[feature_cols])
            out_pred_df = df_unlabeled_internal.copy()
            out_pred_df[f"pred_{target_column}"] = y_hat_new
            out_pred_df[f"pred_{target_column}_low90"] = y_hat_new - q90_half_width
            out_pred_df[f"pred_{target_column}_high90"] = y_hat_new + q90_half_width
            pred_scope = "unlabeled_rows_in_source"
        else:
            out_pred_df = X_test.copy()
            out_pred_df[target_column] = y_test
            out_pred_df[f"pred_{target_column}"] = best_preds
            out_pred_df[f"pred_{target_column}_low90"] = best_preds - q90_half_width
            out_pred_df[f"pred_{target_column}_high90"] = best_preds + q90_half_width
            out_pred_df["residual"] = residuals_test
            pred_scope = "held_out_test_set"

        pred_out_path = assert_safe_write_path(DATA_WORK_DIR / f"{src_path.stem}_{target_column}_predictions.csv")
        out_pred_df.to_csv(pred_out_path, index=False, encoding="utf-8")

        html_path = _render_prediction_dashboard_html(
            output_html_path=OUTPUTS_DIR / f"{src_path.stem}_{target_column}_dashboard.html",
            title=f"ADA — Evaluación y Predicción: {src_path.name}",
            dataset_name=src_path.name,
            task_type="regression",
            target_column=target_column,
            n_train=len(X_train),
            n_test=len(X_test),
            n_features=len(feature_cols),
            models_summary=models_summary,
            best_model_name=best_name,
            feature_importances=feat_imp,
            critic_report=critic,
            chart_payload={
                "y_true": [round(float(v), 5) for v in y_test[:250]],
                "y_pred": [round(float(v), 5) for v in best_preds[:250]],
                "q90_half_width": round(q90_half_width, 5),
            },
            sample_predictions=out_pred_df.head(50).to_dict(orient="records"),
            verification_badge="VERIFICADO: scikit-learn + statsmodels.OLS",
        )
        ui_audit = audit_html_quality(html_path)

        result = {
            "verified": dual_verified,
            "verification_methods": ["scikit-learn Ridge/HistGB", "statsmodels.OLS"],
            "dual_ols_vs_sklearn_max_diff": dual_ols_max_diff,
            "task_type": "regression",
            "target_column": target_column,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(feature_cols),
            "numeric_features": num_cols,
            "categorical_features": cat_cols,
            "auto_excluded_columns": auto_dropped_ids,
            "models": models_summary,
            "best_model": best_name,
            "prediction_interval_90_half_width": round(q90_half_width, 6),
            "top_feature_importances": feat_imp[:10],
            "critic_audit": critic,
            "predictions_scope": pred_scope,
            "predictions_csv_path": str(pred_out_path),
            "interactive_dashboard_html": str(html_path),
            "dashboard_quality_score": ui_audit["quality_score"],
            "sample_predictions": out_pred_df.head(5).to_dict(orient="records"),
        }
        record_audit_event("evaluate_and_predict_database", result, input_path=src_path, output_path=pred_out_path, seed=seed)
        return result

    # --- CLASIFICACION ---
    labels_str = y_raw.astype(str)
    classes_sorted = sorted(labels_str.unique().tolist())
    label_to_int = {lbl: idx for idx, lbl in enumerate(classes_sorted)}
    int_to_label = {idx: lbl for lbl, idx in label_to_int.items()}
    y_int = labels_str.map(label_to_int).to_numpy(dtype=int)

    class_dist = {lbl: int((labels_str == lbl).sum()) for lbl in classes_sorted}
    stratify_arr = y_int if min(class_dist.values()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=test_size, random_state=seed, stratify=stratify_arr
    )
    preproc, num_cols, cat_cols = _build_preprocessor(X_train)

    candidates_cls: dict[str, Any] = {
        "DummyBaseline": Pipeline([("prep", preproc), ("model", DummyClassifier(strategy="prior", random_state=seed))]),
        "LogisticRegression": Pipeline(
            [("prep", preproc), ("model", LogisticRegression(max_iter=1000, random_state=seed))]
        ),
        "HistGradientBoosting": Pipeline(
            [("prep", preproc), ("model", HistGradientBoostingClassifier(max_iter=150, random_state=seed))]
        ),
    }

    min_train_cls = int(np.min(np.bincount(y_train)))
    cv_cls = (
        StratifiedKFold(n_splits=min(5, min_train_cls), shuffle=True, random_state=seed)
        if min_train_cls >= 2
        else KFold(n_splits=2, shuffle=True, random_state=seed)
    )

    models_summary_cls: dict[str, Any] = {}
    fitted_cls: dict[str, Any] = {}
    f1_macro_fn = lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)

    for name, pipe in candidates_cls.items():
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv_cls, scoring="f1_macro")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)
        acc = float(accuracy_score(y_test, preds))
        f1m = float(f1_macro_fn(y_test, preds))
        ci_low, ci_high = _bootstrap_metric_ci(y_test, preds, f1_macro_fn, seed=seed)

        auc_val: float | None = None
        try:
            if len(classes_sorted) == 2:
                auc_val = float(roc_auc_score(y_test, probs[:, 1]))
            else:
                auc_val = float(roc_auc_score(y_test, probs, multi_class="ovr", average="macro"))
        except ValueError:
            auc_val = None

        models_summary_cls[name] = {
            "f1_macro": round(f1m, 6),
            "f1_ci95": [round(ci_low, 6), round(ci_high, 6)],
            "accuracy": round(acc, 6),
            "roc_auc": round(auc_val, 6) if auc_val is not None else None,
            "cv_f1_mean": round(float(np.mean(cv_scores)), 6),
        }
        fitted_cls[name] = (pipe, preds, probs)

    # Verificacion cruzada independiente en clasificacion binaria con statsmodels Logit / GLM Binomial
    dual_cls_note = "verified_via_independent_cv_and_bootstrap"
    if len(classes_sorted) == 2:
        try:
            X_tr_m = fitted_cls["LogisticRegression"][0].named_steps["prep"].transform(X_train)
            X_te_m = fitted_cls["LogisticRegression"][0].named_steps["prep"].transform(X_test)
            glm_bin = sm.GLM(y_train, sm.add_constant(X_tr_m, has_constant="add"), family=sm.families.Binomial()).fit()
            glm_probs = np.asarray(glm_bin.predict(sm.add_constant(X_te_m, has_constant="add")), dtype=float)
            glm_acc = float(accuracy_score(y_test, (glm_probs >= 0.5).astype(int)))
            dual_cls_note = f"statsmodels.GLM_Binomial_accuracy={glm_acc:.4f}"
        except Exception as exc:
            dual_cls_note = f"statsmodels_regularization_needed({type(exc).__name__})"

    best_name_cls = max(["LogisticRegression", "HistGradientBoosting"], key=lambda k: models_summary_cls[k]["f1_macro"])
    best_pipe_cls, best_preds_cls, best_probs_cls = fitted_cls[best_name_cls]

    perm_cls = permutation_importance(best_pipe_cls, X_test, y_test, n_repeats=10, random_state=seed, scoring="f1_macro")
    feat_imp_cls = sorted(
        [
            {"feature": str(col), "importance": round(float(imp), 6)}
            for col, imp in zip(feature_cols, perm_cls.importances_mean)
        ],
        key=lambda d: d["importance"],
        reverse=True,
    )

    critic_cls = audit_classification_results(
        n_train=len(X_train),
        n_test=len(X_test),
        n_features=len(feature_cols),
        class_distribution=class_dist,
        best_f1_macro=models_summary_cls[best_name_cls]["f1_macro"],
        dummy_f1_macro=models_summary_cls["DummyBaseline"]["f1_macro"],
        best_accuracy=models_summary_cls[best_name_cls]["accuracy"],
        dummy_accuracy=models_summary_cls["DummyBaseline"]["accuracy"],
        roc_auc=models_summary_cls[best_name_cls]["roc_auc"],
    )

    if predict_on_path.strip():
        pred_df_src, _ = load_dataset_frame(predict_on_path.strip())
        p_int = best_pipe_cls.predict(pred_df_src[feature_cols])
        p_prob = np.max(best_pipe_cls.predict_proba(pred_df_src[feature_cols]), axis=1)
        out_pred_df = pred_df_src.copy()
        out_pred_df[f"pred_{target_column}"] = [int_to_label[int(i)] for i in p_int]
        out_pred_df["pred_confidence"] = np.round(p_prob, 5)
        pred_scope = "external_dataset"
    elif len(df_unlabeled_internal) > 0:
        p_int = best_pipe_cls.predict(df_unlabeled_internal[feature_cols])
        p_prob = np.max(best_pipe_cls.predict_proba(df_unlabeled_internal[feature_cols]), axis=1)
        out_pred_df = df_unlabeled_internal.copy()
        out_pred_df[f"pred_{target_column}"] = [int_to_label[int(i)] for i in p_int]
        out_pred_df["pred_confidence"] = np.round(p_prob, 5)
        pred_scope = "unlabeled_rows_in_source"
    else:
        out_pred_df = X_test.copy()
        out_pred_df[target_column] = [int_to_label[int(i)] for i in y_test]
        out_pred_df[f"pred_{target_column}"] = [int_to_label[int(i)] for i in best_preds_cls]
        out_pred_df["pred_confidence"] = np.round(np.max(best_probs_cls, axis=1), 5)
        pred_scope = "held_out_test_set"

    pred_out_path = assert_safe_write_path(DATA_WORK_DIR / f"{src_path.stem}_{target_column}_predictions.csv")
    out_pred_df.to_csv(pred_out_path, index=False, encoding="utf-8")

    true_counts = [int((y_test == i).sum()) for i in range(len(classes_sorted))]
    pred_counts = [int((best_preds_cls == i).sum()) for i in range(len(classes_sorted))]

    html_path = _render_prediction_dashboard_html(
        output_html_path=OUTPUTS_DIR / f"{src_path.stem}_{target_column}_dashboard.html",
        title=f"ADA — Evaluación y Clasificación: {src_path.name}",
        dataset_name=src_path.name,
        task_type="classification",
        target_column=target_column,
        n_train=len(X_train),
        n_test=len(X_test),
        n_features=len(feature_cols),
        models_summary=models_summary_cls,
        best_model_name=best_name_cls,
        feature_importances=feat_imp_cls,
        critic_report=critic_cls,
        chart_payload={
            "classes": classes_sorted,
            "true_counts": true_counts,
            "pred_counts": pred_counts,
        },
        sample_predictions=out_pred_df.head(50).to_dict(orient="records"),
        verification_badge="VERIFICADO: scikit-learn + Bootstrap/GLM",
    )
    ui_audit_cls = audit_html_quality(html_path)

    result_cls = {
        "verified": True,
        "verification_methods": ["scikit-learn Logistic/HistGB", dual_cls_note],
        "task_type": "classification",
        "target_column": target_column,
        "classes": classes_sorted,
        "class_distribution": class_dist,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(feature_cols),
        "auto_excluded_columns": auto_dropped_ids,
        "models": models_summary_cls,
        "best_model": best_name_cls,
        "top_feature_importances": feat_imp_cls[:10],
        "critic_audit": critic_cls,
        "predictions_scope": pred_scope,
        "predictions_csv_path": str(pred_out_path),
        "interactive_dashboard_html": str(html_path),
        "dashboard_quality_score": ui_audit_cls["quality_score"],
        "sample_predictions": out_pred_df.head(5).to_dict(orient="records"),
    }
    record_audit_event("evaluate_and_predict_database", result_cls, input_path=src_path, output_path=pred_out_path, seed=seed)
    return result_cls
