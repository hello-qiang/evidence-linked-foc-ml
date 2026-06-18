#!/usr/bin/env python3
"""Stage-5 chemical-family holdout validation and transfer-aware priority tiers."""

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
from sklearn.pipeline import Pipeline


RANDOM_STATE = 20260607
ENDPOINTS = ["LC50", "EC50", "NOEC", "LOEC", "NOEL", "LOEL"]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def make_model(name, seed, n_jobs, n_estimators=320):
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


def calc_metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan
    try:
        rho = spearmanr(y_true, y_pred).correlation
    except Exception:
        rho = np.nan
    return rmse, mae, r2, rho


def reliability_class(rmse, rho):
    if pd.isna(rmse):
        return "insufficient"
    if rmse <= 1.25 and (pd.isna(rho) or rho >= 0.45):
        return "strong_transfer"
    if rmse <= 1.60 and (pd.isna(rho) or rho >= 0.25):
        return "moderate_transfer"
    return "weak_transfer"


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
    return valid, X, numeric_cols


def load_endpoint_labels(base, valid_ids):
    labels = pd.read_csv(os.path.join(base, "07_outputs", "stage2_modeling", "training_labels_long.tsv"), sep="\t")
    labels["target_log10_mg_l"] = pd.to_numeric(labels["target_log10_mg_l"], errors="coerce")
    labels = labels[np.isfinite(labels["target_log10_mg_l"])].copy()
    labels = labels[labels["candidate_id"].astype(str).isin(set(valid_ids))].copy()
    labels = labels[(labels["task_level"] == "endpoint") & (labels["task_label"].isin(ENDPOINTS))].copy()
    return labels


def family_holdout(valid, X, labels, min_test, min_train, n_jobs, n_estimators):
    id_to_idx = {cid: i for i, cid in enumerate(valid["candidate_id"].astype(str))}
    meta_cols = ["candidate_id", "preferred_name", "dtxsid", "casrn", "chemical_family", "split_group"]
    valid_meta = valid[meta_cols].copy()
    metric_rows = []
    pred_rows = []
    for endpoint in ENDPOINTS:
        task = labels[labels["task_label"] == endpoint].drop_duplicates("candidate_id").copy()
        if len(task) < min_train + min_test:
            continue
        task = task.merge(valid_meta, on="candidate_id", how="left")
        family_counts = task["chemical_family"].value_counts()
        families = family_counts[family_counts >= min_test].index.tolist()
        for family in families:
            test_task = task[task["chemical_family"] == family].copy()
            train_task = task[task["chemical_family"] != family].copy()
            if len(test_task) < min_test or len(train_task) < min_train:
                continue
            train_idx = np.array([id_to_idx[str(cid)] for cid in train_task["candidate_id"]], dtype=int)
            test_idx = np.array([id_to_idx[str(cid)] for cid in test_task["candidate_id"]], dtype=int)
            y_train = train_task["target_log10_mg_l"].to_numpy(dtype=float)
            y_test = test_task["target_log10_mg_l"].to_numpy(dtype=float)
            for model_name in ["extra_trees", "random_forest"]:
                seed = RANDOM_STATE + len(endpoint) * 101 + len(family)
                model = make_model(model_name, seed, n_jobs, n_estimators)
                model.fit(X[train_idx], y_train)
                pred = model.predict(X[test_idx])
                rmse, mae, r2, rho = calc_metrics(y_test, pred)
                metric_rows.append(
                    {
                        "task_label": endpoint,
                        "task_id": task["task_id"].iloc[0],
                        "chemical_family": family,
                        "model": model_name,
                        "n_train": len(train_task),
                        "n_test": len(test_task),
                        "train_families": train_task["chemical_family"].nunique(),
                        "test_target_median": float(np.median(y_test)),
                        "pred_median": float(np.median(pred)),
                        "bias_median_pred_minus_obs": float(np.median(pred - y_test)),
                        "rmse_log10": rmse,
                        "mae_log10": mae,
                        "r2": r2,
                        "spearman_rho": rho,
                        "transfer_class": reliability_class(rmse, rho),
                    }
                )
                part = test_task[
                    ["candidate_id", "preferred_name", "dtxsid", "casrn", "chemical_family", "target_log10_mg_l"]
                ].copy()
                part["task_label"] = endpoint
                part["model"] = model_name
                part["family_holdout_pred_log10_mg_l"] = pred
                part["family_holdout_residual"] = pred - y_test
                pred_rows.append(part)
    return pd.DataFrame(metric_rows), pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()


def summarize_transfer(metrics_df):
    if metrics_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    best = (
        metrics_df.sort_values(["task_label", "chemical_family", "rmse_log10", "spearman_rho"], ascending=[True, True, True, False])
        .groupby(["task_label", "chemical_family"])
        .head(1)
        .copy()
    )
    endpoint_summary = best[
        [
            "task_label",
            "chemical_family",
            "model",
            "n_test",
            "rmse_log10",
            "mae_log10",
            "spearman_rho",
            "bias_median_pred_minus_obs",
            "transfer_class",
        ]
    ].copy()
    family_summary = (
        best.groupby("chemical_family")
        .agg(
            endpoints_tested=("task_label", "nunique"),
            total_test_labels=("n_test", "sum"),
            median_rmse_log10=("rmse_log10", "median"),
            median_spearman_rho=("spearman_rho", "median"),
            weak_transfer_endpoints=("transfer_class", lambda s: int((s == "weak_transfer").sum())),
            strong_transfer_endpoints=("transfer_class", lambda s: int((s == "strong_transfer").sum())),
        )
        .reset_index()
    )
    family_summary["family_transfer_class"] = [
        reliability_class(rmse, rho) for rmse, rho in zip(family_summary["median_rmse_log10"], family_summary["median_spearman_rho"])
    ]
    family_summary = family_summary.sort_values(["median_rmse_log10", "median_spearman_rho"], ascending=[True, False])
    return endpoint_summary, family_summary


def priority_context(base, endpoint_summary, family_summary, valid):
    stage4_path = os.path.join(base, "07_outputs", "stage4_robustness_uncertainty", "stage4_priority_uncertainty.tsv")
    if not os.path.exists(stage4_path):
        return pd.DataFrame()
    priority = pd.read_csv(stage4_path, sep="\t", dtype=str).fillna("")
    numeric_cols = [
        "wqp_records",
        "wqp_detected_records",
        "pred_log10_mean",
        "pred_log10_sd",
        "stage4_priority_score",
        "uncertainty_band_log10",
    ]
    for col in numeric_cols:
        priority[col] = pd.to_numeric(priority[col], errors="coerce")
    family_map = valid[["candidate_id", "chemical_family"]].copy()
    out = priority.merge(family_map, on="candidate_id", how="left")
    out = out.merge(
        endpoint_summary.rename(
            columns={
                "rmse_log10": "family_endpoint_rmse_log10",
                "spearman_rho": "family_endpoint_spearman_rho",
                "transfer_class": "family_endpoint_transfer_class",
                "model": "family_endpoint_model",
            }
        )[["task_label", "chemical_family", "family_endpoint_model", "family_endpoint_rmse_log10", "family_endpoint_spearman_rho", "family_endpoint_transfer_class"]],
        on=["task_label", "chemical_family"],
        how="left",
    )
    out = out.merge(
        family_summary[["chemical_family", "family_transfer_class", "median_rmse_log10", "median_spearman_rho", "endpoints_tested"]],
        on="chemical_family",
        how="left",
    )
    def tier(row):
        pconf = row.get("priority_confidence", "")
        eclass = row.get("family_endpoint_transfer_class", "")
        fclass = row.get("family_transfer_class", "")
        score = row.get("stage4_priority_score", np.nan)
        if pconf == "high" and eclass in ("strong_transfer", "moderate_transfer") and score >= 4.0:
            return "Tier 1: high-confidence monitoring target"
        if pconf in ("high", "medium") and fclass != "weak_transfer" and score >= 3.5:
            return "Tier 2: priority with review"
        if pconf == "low" or eclass == "weak_transfer":
            return "Tier 3: uncertainty-driven follow-up"
        return "Tier 4: lower immediate priority"
    out["transfer_aware_action_tier"] = out.apply(tier, axis=1)
    return out.sort_values(["transfer_aware_action_tier", "stage4_priority_score"], ascending=[True, False])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=os.path.expanduser("~/evidence_linked_foc_project"))
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    parser.add_argument("--min-test", type=int, default=15)
    parser.add_argument("--min-train", type=int, default=80)
    parser.add_argument("--trees", type=int, default=320)
    args = parser.parse_args(argv)
    base = args.base
    out_dir = os.path.join(base, "07_outputs", "stage5_family_transfer")
    ensure_dir(out_dir)
    print("started_utc=%s" % now())
    print("base=%s" % base)
    valid, X, numeric_cols = load_feature_layer(base)
    labels = load_endpoint_labels(base, valid["candidate_id"].astype(str))
    metrics_df, predictions = family_holdout(valid, X, labels, args.min_test, args.min_train, args.n_jobs, args.trees)
    endpoint_summary, family_summary = summarize_transfer(metrics_df)
    priority = priority_context(base, endpoint_summary, family_summary, valid)
    metrics_df.to_csv(os.path.join(out_dir, "stage5_family_holdout_metrics.tsv"), sep="\t", index=False)
    predictions.to_csv(os.path.join(out_dir, "stage5_family_holdout_predictions.tsv"), sep="\t", index=False)
    endpoint_summary.to_csv(os.path.join(out_dir, "stage5_endpoint_family_transfer_summary.tsv"), sep="\t", index=False)
    family_summary.to_csv(os.path.join(out_dir, "stage5_family_transfer_summary.tsv"), sep="\t", index=False)
    priority.to_csv(os.path.join(out_dir, "stage5_transfer_aware_priority.tsv"), sep="\t", index=False)
    summary = {
        "created_utc": now(),
        "candidate_rows": int(len(valid)),
        "feature_columns": int(X.shape[1]),
        "numeric_descriptor_columns": int(len(numeric_cols)),
        "endpoint_label_rows": int(len(labels)),
        "metric_rows": int(len(metrics_df)),
        "holdout_prediction_rows": int(len(predictions)),
        "endpoint_family_summary_rows": int(len(endpoint_summary)),
        "family_summary_rows": int(len(family_summary)),
        "transfer_aware_priority_rows": int(len(priority)),
        "min_test": int(args.min_test),
        "min_train": int(args.min_train),
        "trees_per_model": int(args.trees),
        "note": "Chemical-family holdout validation and transfer-aware action tiers for uncertainty-aware priority candidates.",
    }
    with open(os.path.join(out_dir, "stage5_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with open(os.path.join(base, "01_download_logs", "processing", "stage5_family_transfer_summary.txt"), "w", encoding="utf-8") as handle:
        for key in sorted(summary):
            handle.write("%s=%s\n" % (key, summary[key]))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("finished_utc=%s" % now())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
