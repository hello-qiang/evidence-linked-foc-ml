#!/usr/bin/env python3
"""Stage-4 repeated scaffold validation and uncertainty-aware prioritization."""

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline


RANDOM_STATE = 20260607
ENDPOINTS = ["LC50", "EC50", "NOEC", "LOEC", "NOEL", "LOEL"]
ACUTE_ENDPOINTS = ["LC50", "EC50"]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def make_model(name, seed, n_jobs, n_estimators=300):
    if name == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=n_jobs,
        )
    elif name == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=n_jobs,
        )
    else:
        raise ValueError("unknown model: %s" % name)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])


def metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan
    try:
        rho = spearmanr(y_true, y_pred).correlation
    except Exception:
        rho = np.nan
    return rmse, mae, r2, rho


def load_feature_layer(base):
    stage3 = os.path.join(base, "07_outputs", "stage3_rdkit_modeling")
    descriptors = pd.read_csv(os.path.join(stage3, "candidate_rdkit_descriptors.tsv"), sep="\t", dtype=str).fillna("")
    valid_ids = np.load(os.path.join(stage3, "candidate_morgan1024_valid_ids.npy"), allow_pickle=False).astype(str)
    fp_matrix = np.load(os.path.join(stage3, "candidate_morgan1024.npy")).astype(np.float32)
    valid = descriptors.set_index("candidate_id").loc[valid_ids].reset_index()
    exclude = {
        "candidate_id",
        "smiles",
        "preferred_name",
        "casrn",
        "dtxsid",
        "pubchem_cid",
        "inchikey",
        "canonical_smiles",
        "isomeric_canonical_smiles",
        "chemical_family",
        "murcko_scaffold",
        "split_group",
        "stage1_data_status",
    }
    numeric_cols = [
        col
        for col in valid.columns
        if col not in exclude and (col.startswith("rdkit_") or col == "has_fluorine")
    ]
    numeric = valid[numeric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    X = np.hstack([numeric, fp_matrix]).astype(np.float32)
    feature_names = numeric_cols + ["morgan_%04d" % i for i in range(fp_matrix.shape[1])]
    return valid, X, feature_names, numeric_cols


def load_endpoint_labels(base, valid_ids):
    labels = pd.read_csv(os.path.join(base, "07_outputs", "stage2_modeling", "training_labels_long.tsv"), sep="\t")
    labels["target_log10_mg_l"] = pd.to_numeric(labels["target_log10_mg_l"], errors="coerce")
    labels = labels[np.isfinite(labels["target_log10_mg_l"])].copy()
    labels = labels[labels["candidate_id"].astype(str).isin(set(valid_ids))].copy()
    labels = labels[(labels["task_level"] == "endpoint") & (labels["task_label"].isin(ENDPOINTS))].copy()
    return labels


def repeated_scaffold_metrics(valid, X, labels, repeats, n_jobs, n_estimators):
    id_to_idx = {cid: i for i, cid in enumerate(valid["candidate_id"].astype(str))}
    rows = []
    for endpoint in ENDPOINTS:
        task = labels[labels["task_label"] == endpoint].drop_duplicates("candidate_id").copy()
        if len(task) < 80:
            continue
        idx = np.array([id_to_idx[str(cid)] for cid in task["candidate_id"]], dtype=int)
        y = task["target_log10_mg_l"].to_numpy(dtype=float)
        groups = valid.loc[idx, "split_group"].fillna("missing_group").astype(str).to_numpy()
        unique_groups = len(set(groups))
        for repeat in range(repeats):
            seed = RANDOM_STATE + repeat * 101 + len(endpoint)
            if unique_groups >= 8:
                splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
                local_train, local_test = next(splitter.split(idx, y, groups=groups))
                train_idx, test_idx = idx[local_train], idx[local_test]
                y_train, y_test = y[local_train], y[local_test]
                split_name = "scaffold_group"
            else:
                train_idx, test_idx, y_train, y_test = train_test_split(idx, y, test_size=0.25, random_state=seed)
                split_name = "random_fallback"
            for model_name in ["extra_trees", "random_forest"]:
                for control in ["observed", "permuted"]:
                    model = make_model(model_name, seed, n_jobs, n_estimators=n_estimators)
                    fit_y = y_train.copy()
                    if control == "permuted":
                        rng = np.random.default_rng(seed)
                        fit_y = rng.permutation(fit_y)
                    model.fit(X[train_idx], fit_y)
                    pred = model.predict(X[test_idx])
                    rmse, mae, r2, rho = metrics(y_test, pred)
                    rows.append(
                        {
                            "task_label": endpoint,
                            "task_id": task["task_id"].iloc[0],
                            "split": split_name,
                            "repeat": repeat,
                            "model": model_name,
                            "control": control,
                            "n_candidates": len(task),
                            "n_groups": unique_groups,
                            "n_train": len(train_idx),
                            "n_test": len(test_idx),
                            "rmse_log10": rmse,
                            "mae_log10": mae,
                            "r2": r2,
                            "spearman_rho": rho,
                        }
                    )
    return pd.DataFrame(rows)


def choose_models(metric_df):
    observed = metric_df[metric_df["control"] == "observed"].copy()
    summary = (
        observed.groupby(["task_label", "model"])
        .agg(
            median_rmse_log10=("rmse_log10", "median"),
            median_spearman_rho=("spearman_rho", "median"),
            q25_rmse_log10=("rmse_log10", lambda s: s.quantile(0.25)),
            q75_rmse_log10=("rmse_log10", lambda s: s.quantile(0.75)),
            repeats=("rmse_log10", "size"),
        )
        .reset_index()
    )
    best = (
        summary.sort_values(["task_label", "median_rmse_log10", "median_spearman_rho"], ascending=[True, True, False])
        .groupby("task_label")
        .head(1)
    )
    return summary, dict(zip(best["task_label"], best["model"]))


def bootstrap_uncertainty(valid, X, feature_names, labels, best_models, bootstraps, n_jobs, n_estimators):
    id_to_idx = {cid: i for i, cid in enumerate(valid["candidate_id"].astype(str))}
    rows = []
    importance_rows = []
    metadata_cols = [
        "candidate_id",
        "preferred_name",
        "casrn",
        "dtxsid",
        "pubchem_cid",
        "inchikey",
        "has_fluorine",
        "chemical_family",
        "split_group",
    ]
    meta = valid[metadata_cols].copy()
    for endpoint in ENDPOINTS:
        task = labels[labels["task_label"] == endpoint].drop_duplicates("candidate_id").copy()
        if len(task) < 80:
            continue
        model_name = best_models.get(endpoint, "random_forest")
        idx = np.array([id_to_idx[str(cid)] for cid in task["candidate_id"]], dtype=int)
        y = task["target_log10_mg_l"].to_numpy(dtype=float)
        preds = []
        rng = np.random.default_rng(RANDOM_STATE + 7919 + len(endpoint))
        for boot in range(bootstraps):
            seed = RANDOM_STATE + 1009 * boot + 17 * len(endpoint)
            sample_local = rng.integers(0, len(idx), len(idx))
            train_idx = idx[sample_local]
            train_y = y[sample_local]
            model = make_model(model_name, seed, n_jobs, n_estimators=n_estimators)
            model.fit(X[train_idx], train_y)
            preds.append(model.predict(X))
        pred = np.vstack(preds)
        task_rows = meta.copy()
        task_rows["task_label"] = endpoint
        task_rows["task_id"] = task["task_id"].iloc[0]
        task_rows["bootstrap_model"] = model_name
        task_rows["train_candidates"] = len(task)
        task_rows["pred_log10_mean"] = pred.mean(axis=0)
        task_rows["pred_log10_sd"] = pred.std(axis=0, ddof=1) if bootstraps > 1 else 0.0
        task_rows["pred_log10_q05"] = np.quantile(pred, 0.05, axis=0)
        task_rows["pred_log10_q50"] = np.quantile(pred, 0.50, axis=0)
        task_rows["pred_log10_q95"] = np.quantile(pred, 0.95, axis=0)
        task_rows["pred_mg_l_q50"] = np.power(10.0, task_rows["pred_log10_q50"])
        rows.append(task_rows)
        full_model = make_model(model_name, RANDOM_STATE + 313 + len(endpoint), n_jobs, n_estimators=n_estimators)
        full_model.fit(X[idx], y)
        importances = full_model.named_steps["model"].feature_importances_
        order = np.argsort(importances)[::-1][:80]
        for rank, pos in enumerate(order, start=1):
            importance_rows.append(
                {
                    "task_label": endpoint,
                    "model": model_name,
                    "rank": rank,
                    "feature": "morgan_fingerprint_bits" if str(feature_names[pos]).startswith("morgan_") else feature_names[pos],
                    "feature_raw": feature_names[pos],
                    "importance": float(importances[pos]),
                }
            )
    uncertainty = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    importance = pd.DataFrame(importance_rows)
    if len(importance):
        importance = (
            importance.groupby(["task_label", "model", "feature"])
            .agg(total_importance=("importance", "sum"), best_rank=("rank", "min"))
            .reset_index()
            .sort_values(["task_label", "total_importance"], ascending=[True, False])
        )
    return uncertainty, importance


def priority_with_uncertainty(base, uncertainty):
    matrix = pd.read_csv(os.path.join(base, "07_outputs", "stage1_data_package", "candidate_stage1_matrix.tsv"), sep="\t", dtype=str).fillna("")
    stage3_pred_path = os.path.join(base, "07_outputs", "stage3_rdkit_modeling", "rdkit_task_predictions.tsv")
    ad = pd.read_csv(
        stage3_pred_path,
        sep="\t",
        usecols=["candidate_id", "task_id", "rdkit_ad_pass"],
        dtype={"candidate_id": str, "task_id": str},
    )
    ad["rdkit_ad_pass"] = pd.to_numeric(ad["rdkit_ad_pass"], errors="coerce").fillna(0).astype(int)
    acute = uncertainty[uncertainty["task_label"].isin(ACUTE_ENDPOINTS)].merge(ad, on=["candidate_id", "task_id"], how="left")
    acute = acute[pd.to_numeric(acute["rdkit_ad_pass"], errors="coerce").fillna(0).astype(int) == 1].copy()
    if acute.empty:
        return pd.DataFrame()
    acute = acute.sort_values(["candidate_id", "pred_log10_mean"]).groupby("candidate_id").head(1)
    base_cols = [
        "candidate_id",
        "preferred_name",
        "casrn",
        "dtxsid",
        "pubchem_cid",
        "inchikey",
        "has_fluorine",
        "wqp_records",
        "wqp_detected_records",
        "wqp_detected_fraction",
        "stage1_data_status",
    ]
    out = matrix[base_cols].merge(
        acute[
            [
                "candidate_id",
                "task_id",
                "task_label",
                "bootstrap_model",
                "train_candidates",
                "pred_log10_mean",
                "pred_log10_sd",
                "pred_log10_q05",
                "pred_log10_q50",
                "pred_log10_q95",
                "pred_mg_l_q50",
            ]
        ],
        on="candidate_id",
        how="inner",
    )
    for col in ["has_fluorine", "wqp_records", "wqp_detected_records", "wqp_detected_fraction"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    out = out[out["wqp_records"] > 0].copy()
    out["exposure_score"] = np.log10(out["wqp_detected_records"] + 1.0) + 0.2 * np.log10(out["wqp_records"] + 1.0)
    out["uncertainty_band_log10"] = out["pred_log10_q95"] - out["pred_log10_q05"]
    out["toxicity_score_mean"] = -out["pred_log10_mean"]
    out["stage4_priority_score"] = (
        out["toxicity_score_mean"]
        + out["exposure_score"]
        + 0.25 * out["has_fluorine"]
        - 0.50 * out["pred_log10_sd"]
    )
    out["priority_confidence"] = np.where(
        out["pred_log10_sd"] <= 0.35,
        "high",
        np.where(out["pred_log10_sd"] <= 0.75, "medium", "low"),
    )
    return out.sort_values("stage4_priority_score", ascending=False)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=os.path.expanduser("~/evidence_linked_foc_project"))
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--bootstraps", type=int, default=40)
    parser.add_argument("--trees", type=int, default=260)
    args = parser.parse_args(argv)
    base = args.base
    out_dir = os.path.join(base, "07_outputs", "stage4_robustness_uncertainty")
    ensure_dir(out_dir)
    print("started_utc=%s" % now())
    print("base=%s" % base)
    valid, X, feature_names, numeric_cols = load_feature_layer(base)
    labels = load_endpoint_labels(base, valid["candidate_id"].astype(str))
    metric_df = repeated_scaffold_metrics(valid, X, labels, args.repeats, args.n_jobs, args.trees)
    model_summary, best_models = choose_models(metric_df)
    uncertainty, importance = bootstrap_uncertainty(valid, X, feature_names, labels, best_models, args.bootstraps, args.n_jobs, args.trees)
    priority = priority_with_uncertainty(base, uncertainty)
    metric_df.to_csv(os.path.join(out_dir, "stage4_repeated_scaffold_metrics.tsv"), sep="\t", index=False)
    model_summary.to_csv(os.path.join(out_dir, "stage4_model_stability_summary.tsv"), sep="\t", index=False)
    uncertainty.to_csv(os.path.join(out_dir, "stage4_prediction_uncertainty.tsv"), sep="\t", index=False)
    importance.to_csv(os.path.join(out_dir, "stage4_descriptor_importance.tsv"), sep="\t", index=False)
    priority.to_csv(os.path.join(out_dir, "stage4_priority_uncertainty.tsv"), sep="\t", index=False)
    observed = metric_df[metric_df["control"] == "observed"].copy()
    permuted = metric_df[metric_df["control"] == "permuted"].copy()
    summary = {
        "created_utc": now(),
        "candidate_rows": int(len(valid)),
        "feature_columns": int(X.shape[1]),
        "numeric_descriptor_columns": int(len(numeric_cols)),
        "endpoint_label_rows": int(len(labels)),
        "endpoints": sorted(labels["task_label"].unique().tolist()),
        "repeats": int(args.repeats),
        "bootstraps": int(args.bootstraps),
        "trees_per_model": int(args.trees),
        "metric_rows": int(len(metric_df)),
        "uncertainty_rows": int(len(uncertainty)),
        "priority_rows": int(len(priority)),
        "observed_median_rmse_log10": float(observed["rmse_log10"].median()) if len(observed) else None,
        "permuted_median_rmse_log10": float(permuted["rmse_log10"].median()) if len(permuted) else None,
        "best_models": best_models,
        "note": "Repeated scaffold/family validation, label-permutation controls, bootstrap prediction intervals, and uncertainty-aware exposed priority scores.",
    }
    with open(os.path.join(out_dir, "stage4_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with open(os.path.join(base, "01_download_logs", "processing", "stage4_robustness_uncertainty_summary.txt"), "w", encoding="utf-8") as handle:
        for key in sorted(summary):
            handle.write("%s=%s\n" % (key, summary[key]))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("finished_utc=%s" % now())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
