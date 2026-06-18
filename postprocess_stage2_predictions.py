#!/usr/bin/env python3
"""Screen stage-2 predictions with a simple applicability-domain layer."""

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd


RANGE_COLS = ["exact_mass", "formula_heavy_atoms", "smiles_len", "formula_halogens"]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_domain_tables(descriptors, labels):
    descriptors = numeric(descriptors.copy(), RANGE_COLS)
    merged = labels.merge(descriptors[["candidate_id"] + RANGE_COLS], on="candidate_id", how="left")
    rows = []
    for task_id, group in merged.groupby("task_id"):
        row = {"task_id": task_id}
        for col in RANGE_COLS:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            if len(values):
                q01, q05, q50, q95, q99 = values.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
                spread = max(q95 - q05, 1.0)
                row[col + "_q01"] = q01
                row[col + "_q05"] = q05
                row[col + "_q50"] = q50
                row[col + "_q95"] = q95
                row[col + "_q99"] = q99
                row[col + "_lower"] = max(0.0, q01 - 0.20 * spread)
                row[col + "_upper"] = q99 + 0.20 * spread
            else:
                for suffix in ["q01", "q05", "q50", "q95", "q99", "lower", "upper"]:
                    row[col + "_" + suffix] = np.nan
        y = pd.to_numeric(group["target_log10_mg_l"], errors="coerce").dropna()
        row["target_q01"] = y.quantile(0.01) if len(y) else np.nan
        row["target_q99"] = y.quantile(0.99) if len(y) else np.nan
        row["target_min"] = y.min() if len(y) else np.nan
        row["target_max"] = y.max() if len(y) else np.nan
        row["training_candidates"] = group["candidate_id"].nunique()
        rows.append(row)
    return pd.DataFrame(rows)


def screen_predictions(predictions, descriptors, domains):
    pred = predictions.copy()
    pred = numeric(pred, ["pred_log10_mg_l", "pred_mg_l", "test_rmse_log10"])
    desc_cols = ["candidate_id"] + RANGE_COLS + ["has_smiles", "has_formula"]
    pred = pred.merge(descriptors[desc_cols], on="candidate_id", how="left", suffixes=("", "_desc"))
    pred = pred.merge(domains, on="task_id", how="left")
    pred = numeric(pred, RANGE_COLS)
    pass_cols = []
    for col in RANGE_COLS:
        pass_col = col + "_ad_pass"
        pass_cols.append(pass_col)
        pred[pass_col] = (
            pd.to_numeric(pred[col], errors="coerce").notna()
            & (pred[col] >= pred[col + "_lower"])
            & (pred[col] <= pred[col + "_upper"])
        ).astype(int)
    pred["ad_score_0_1"] = pred[pass_cols].mean(axis=1)
    pred["ad_pass_basic"] = (pred["ad_score_0_1"] >= 0.75).astype(int)
    pred["pred_log10_mg_l_screened"] = pred["pred_log10_mg_l"].clip(lower=pred["target_q01"], upper=pred["target_q99"])
    pred["pred_mg_l_screened"] = np.power(10.0, pred["pred_log10_mg_l_screened"])
    pred["extrapolation_delta_log10"] = pred["pred_log10_mg_l"] - pred["pred_log10_mg_l_screened"]
    keep = [
        "task_id",
        "task_level",
        "task_label",
        "best_model",
        "train_candidates",
        "test_rmse_log10",
        "candidate_id",
        "preferred_name",
        "casrn",
        "dtxsid",
        "pubchem_cid",
        "inchikey",
        "has_fluorine",
        "has_smiles",
        "has_formula",
        "pred_log10_mg_l",
        "pred_mg_l",
        "pred_log10_mg_l_screened",
        "pred_mg_l_screened",
        "ad_score_0_1",
        "ad_pass_basic",
        "extrapolation_delta_log10",
    ]
    return pred[keep]


def build_priority(screened, matrix):
    pred = screened.copy()
    pred["endpoint"] = pred["task_label"].astype(str).str.split("|").str[0]
    pred["is_acute"] = pred["endpoint"].isin(["LC50", "EC50"])
    acute = pred[(pred["is_acute"]) & (pred["ad_pass_basic"] == 1)].copy()
    if acute.empty:
        return pd.DataFrame(), pd.DataFrame()
    acute_best = (
        acute.sort_values(["candidate_id", "pred_log10_mg_l_screened", "test_rmse_log10"])
        .groupby("candidate_id")
        .head(1)
        .rename(columns={"pred_log10_mg_l_screened": "pred_min_acute_screened_log10_mg_l"})
    )
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
    base = matrix[base_cols].copy()
    for col in ["has_fluorine", "wqp_records", "wqp_detected_records", "wqp_detected_fraction"]:
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0)
    out = base.merge(
        acute_best[
            [
                "candidate_id",
                "task_id",
                "task_label",
                "best_model",
                "train_candidates",
                "test_rmse_log10",
                "pred_min_acute_screened_log10_mg_l",
                "pred_mg_l_screened",
                "ad_score_0_1",
            ]
        ],
        on="candidate_id",
        how="inner",
    )
    out["exposure_score"] = np.log10(out["wqp_detected_records"] + 1.0) + 0.2 * np.log10(out["wqp_records"] + 1.0)
    out["toxicity_score_screened"] = -out["pred_min_acute_screened_log10_mg_l"]
    out["priority_score_screened"] = (
        out["toxicity_score_screened"]
        + out["exposure_score"]
        + 0.25 * out["has_fluorine"]
        - 0.25 * pd.to_numeric(out["test_rmse_log10"], errors="coerce").fillna(1.5)
    )
    exposed = out[out["wqp_records"] > 0].sort_values("priority_score_screened", ascending=False)
    all_screened = out.sort_values("priority_score_screened", ascending=False)
    return exposed, all_screened


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=os.path.expanduser("~/evidence_linked_foc_project"))
    args = parser.parse_args(argv)
    base = args.base
    out_dir = os.path.join(base, "07_outputs", "stage2_modeling")
    matrix = pd.read_csv(os.path.join(base, "07_outputs", "stage1_data_package", "candidate_stage1_matrix.tsv"), sep="\t", dtype=str).fillna("")
    descriptors = pd.read_csv(os.path.join(out_dir, "candidate_descriptors.tsv"), sep="\t", dtype=str).fillna("")
    labels = pd.read_csv(os.path.join(out_dir, "training_labels_long.tsv"), sep="\t")
    predictions = pd.read_csv(os.path.join(out_dir, "baseline_task_predictions.tsv"), sep="\t")
    domains = build_domain_tables(descriptors, labels)
    screened = screen_predictions(predictions, descriptors, domains)
    exposed_priority, all_priority = build_priority(screened, matrix)
    domains.to_csv(os.path.join(out_dir, "applicability_domain_by_task.tsv"), sep="\t", index=False)
    screened.to_csv(os.path.join(out_dir, "baseline_task_predictions_screened.tsv"), sep="\t", index=False)
    exposed_priority.to_csv(os.path.join(out_dir, "candidate_priority_exposed_screened.tsv"), sep="\t", index=False)
    all_priority.to_csv(os.path.join(out_dir, "candidate_priority_all_screened.tsv"), sep="\t", index=False)
    summary = {
        "created_utc": now(),
        "screened_prediction_rows": int(len(screened)),
        "ad_pass_prediction_rows": int((screened["ad_pass_basic"] == 1).sum()),
        "domain_task_rows": int(len(domains)),
        "exposed_priority_rows": int(len(exposed_priority)),
        "all_priority_rows": int(len(all_priority)),
        "note": "Use exposed_screened for primary reporting prioritization; raw priority is retained only as an extrapolation audit.",
    }
    with open(os.path.join(out_dir, "stage2_postprocess_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
