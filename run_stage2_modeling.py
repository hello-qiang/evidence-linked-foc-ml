#!/usr/bin/env python3
"""Stage-2 feature building and baseline modeling.

This script uses the stage-1 data package and avoids RDKit so it can run on the
available Anaconda module. Molecular structure is represented by formula/string
descriptors plus hashed SMILES character n-grams.
"""

import argparse
import json
import math
import os
import re
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 20260607
ELEMENTS = ["C", "H", "N", "O", "F", "Cl", "Br", "I", "S", "P", "B", "Si", "Na", "K"]
ENDPOINT_ORDER = ["LC50", "EC50", "NOEC", "LOEC", "NOEL", "LOEL", "EC10", "IC50"]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def slugify(value):
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(value))
    return value.strip("_")[:140]


def to_float(value):
    try:
        if pd.isna(value):
            return np.nan
        text = str(value).strip().replace(",", "")
        if not text:
            return np.nan
        return float(text)
    except Exception:
        return np.nan


def parse_formula_counts(formula):
    counts = {element: 0.0 for element in ELEMENTS}
    if not isinstance(formula, str) or not formula.strip():
        return counts
    for element, number in re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula):
        if element not in counts:
            continue
        amount = 1.0 if number == "" else to_float(number)
        if math.isfinite(amount):
            counts[element] += amount
    return counts


def smiles_metrics(smiles):
    smiles = smiles if isinstance(smiles, str) else ""
    metrics = {
        "smiles_len": len(smiles),
        "smiles_upper_F": smiles.count("F"),
        "smiles_Cl": smiles.count("Cl"),
        "smiles_Br": smiles.count("Br"),
        "smiles_I": smiles.count("I"),
        "smiles_N": smiles.count("N") + smiles.count("n"),
        "smiles_O": smiles.count("O") + smiles.count("o"),
        "smiles_S": smiles.count("S") + smiles.count("s"),
        "smiles_P": smiles.count("P") + smiles.count("p"),
        "smiles_ring_digits": sum(ch.isdigit() for ch in smiles),
        "smiles_branches": smiles.count("(") + smiles.count(")"),
        "smiles_double_bonds": smiles.count("="),
        "smiles_triple_bonds": smiles.count("#"),
        "smiles_brackets": smiles.count("[") + smiles.count("]"),
        "smiles_aromatic_chars": sum(ch in "bcnops" for ch in smiles),
        "smiles_charge_marks": smiles.count("+") + smiles.count("-"),
        "smiles_stereo_marks": smiles.count("@") + smiles.count("/") + smiles.count("\\"),
        "smiles_fragments": smiles.count(".") + (1 if smiles else 0),
    }
    metrics["smiles_halogen_count"] = (
        metrics["smiles_upper_F"] + metrics["smiles_Cl"] + metrics["smiles_Br"] + metrics["smiles_I"]
    )
    metrics["smiles_hetero_count"] = (
        metrics["smiles_N"] + metrics["smiles_O"] + metrics["smiles_S"] + metrics["smiles_P"]
    )
    metrics["smiles_branch_density"] = metrics["smiles_branches"] / max(metrics["smiles_len"], 1)
    metrics["smiles_halogen_density"] = metrics["smiles_halogen_count"] / max(metrics["smiles_len"], 1)
    return metrics


def build_candidate_descriptors(matrix):
    rows = []
    for _, row in matrix.iterrows():
        formula = str(row.get("formula", "") or "")
        smiles = str(row.get("smiles", "") or "")
        counts = parse_formula_counts(formula)
        metrics = smiles_metrics(smiles)
        heavy_atoms = counts["C"] + counts["N"] + counts["O"] + counts["F"] + counts["Cl"] + counts["Br"] + counts["I"] + counts["S"] + counts["P"] + counts["B"] + counts["Si"]
        hetero_atoms = counts["N"] + counts["O"] + counts["S"] + counts["P"] + counts["B"] + counts["Si"]
        halogens = counts["F"] + counts["Cl"] + counts["Br"] + counts["I"]
        desc = OrderedDict()
        for key in [
            "candidate_id",
            "preferred_name",
            "casrn",
            "dtxsid",
            "pubchem_cid",
            "inchikey",
            "smiles",
            "formula",
            "source_flags",
            "evidence_flags",
            "stage1_data_status",
        ]:
            desc[key] = row.get(key, "")
        desc["exact_mass"] = to_float(row.get("pubchem_exact_mass", ""))
        desc["has_pubchem"] = int(row.get("has_pubchem", 0) or 0)
        desc["has_inchikey"] = int(row.get("has_inchikey", 0) or 0)
        desc["has_smiles"] = int(row.get("has_smiles", 0) or 0)
        desc["has_formula"] = int(row.get("has_formula", 0) or 0)
        desc["has_casrn"] = int(row.get("has_casrn", 0) or 0)
        desc["has_dtxsid"] = int(row.get("has_dtxsid", 0) or 0)
        desc["has_fluorine"] = int(row.get("has_fluorine", 0) or 0)
        desc["identifier_completeness_0_6"] = to_float(row.get("identifier_completeness_0_6", ""))
        for element in ELEMENTS:
            desc["formula_%s" % element] = counts[element]
        desc["formula_heavy_atoms"] = heavy_atoms
        desc["formula_hetero_atoms"] = hetero_atoms
        desc["formula_halogens"] = halogens
        desc["formula_f_fraction_heavy"] = counts["F"] / heavy_atoms if heavy_atoms else 0.0
        desc["formula_halogen_fraction_heavy"] = halogens / heavy_atoms if heavy_atoms else 0.0
        desc["formula_hetero_fraction_heavy"] = hetero_atoms / heavy_atoms if heavy_atoms else 0.0
        desc.update(metrics)
        rows.append(desc)
    return pd.DataFrame(rows)


def load_stage1(base):
    package = os.path.join(base, "07_outputs", "stage1_data_package")
    matrix = pd.read_csv(os.path.join(package, "candidate_stage1_matrix.tsv"), sep="\t", dtype=str).fillna("")
    labels = pd.read_csv(os.path.join(package, "ecotox_endpoint_labels_mgL.tsv"), sep="\t", dtype=str).fillna("")
    labels["target_log10_mg_l"] = pd.to_numeric(labels["log10_mg_l_median"], errors="coerce")
    labels["records_mg_l"] = pd.to_numeric(labels["records_mg_l"], errors="coerce").fillna(0).astype(int)
    labels = labels[np.isfinite(labels["target_log10_mg_l"])]
    labels = labels[(labels["target_log10_mg_l"] > -12) & (labels["target_log10_mg_l"] < 8)]
    return matrix, labels


def add_task_rows(labels, task_level, group_cols, min_candidates):
    out = []
    for keys, group in labels.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        task_label = "|".join(str(k) for k in keys)
        task_id = "%s__%s" % (task_level, slugify(task_label))
        candidate_rows = (
            group.groupby("candidate_id")
            .agg(
                target_log10_mg_l=("target_log10_mg_l", "median"),
                label_source_rows=("candidate_id", "size"),
                records_mg_l=("records_mg_l", "sum"),
                endpoint=("endpoint", "first"),
                effect=("effect", lambda s: "|".join(sorted(set(str(x) for x in s if str(x))))),
                ecotox_group=("ecotox_group", lambda s: "|".join(sorted(set(str(x) for x in s if str(x))))[:500]),
                endpoint_family=("endpoint_family", lambda s: "|".join(sorted(set(str(x) for x in s if str(x))))),
                high_support_rows=("label_quality", lambda s: int((s == "high_support").sum())),
                usable_rows=("label_quality", lambda s: int(s.isin(["high_support", "usable_sparse"]).sum())),
            )
            .reset_index()
        )
        n_candidates = candidate_rows["candidate_id"].nunique()
        if n_candidates < min_candidates:
            continue
        candidate_rows["task_id"] = task_id
        candidate_rows["task_level"] = task_level
        candidate_rows["task_label"] = task_label
        candidate_rows["task_candidates"] = n_candidates
        out.append(candidate_rows)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def build_training_labels(labels, descriptor_ids):
    selected_parts = [
        add_task_rows(labels[labels["endpoint"].isin(ENDPOINT_ORDER)], "endpoint", ["endpoint"], 150),
        add_task_rows(labels[labels["endpoint"].isin(ENDPOINT_ORDER)], "endpoint_effect", ["endpoint", "effect"], 120),
        add_task_rows(labels[labels["endpoint"].isin(["LC50", "EC50", "NOEC", "LOEC"])], "endpoint_effect_group", ["endpoint", "effect", "ecotox_group"], 180),
    ]
    selected_parts = [part for part in selected_parts if len(part)]
    tasks = pd.concat(selected_parts, ignore_index=True)
    tasks = tasks[tasks["candidate_id"].isin(descriptor_ids)].copy()
    counts = (
        tasks.groupby(["task_id", "task_level", "task_label"])
        .agg(n_candidates=("candidate_id", "nunique"), rows=("candidate_id", "size"), median_target=("target_log10_mg_l", "median"))
        .reset_index()
        .sort_values(["task_level", "n_candidates"], ascending=[True, False])
    )
    counts = counts[counts["n_candidates"] >= 80].copy()
    tasks = tasks[tasks["task_id"].isin(counts["task_id"])].copy()
    return tasks, counts


def build_feature_matrix(descriptors, hash_features=512):
    text = descriptors["smiles"].fillna("").astype(str).tolist()
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        n_features=hash_features,
        alternate_sign=False,
        norm=None,
        lowercase=False,
    )
    hashed = vectorizer.transform(text).toarray().astype(np.float32)
    exclude = {
        "candidate_id",
        "preferred_name",
        "casrn",
        "dtxsid",
        "pubchem_cid",
        "inchikey",
        "smiles",
        "formula",
        "source_flags",
        "evidence_flags",
        "stage1_data_status",
    }
    numeric_cols = [col for col in descriptors.columns if col not in exclude]
    numeric = descriptors[numeric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    features = np.hstack([numeric, hashed]).astype(np.float32)
    return features, numeric_cols, hash_features


def make_models(n_jobs):
    return {
        "ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=400,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        random_state=RANDOM_STATE,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        random_state=RANDOM_STATE,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        ),
        "mlp": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(96, 48),
                        alpha=0.001,
                        learning_rate_init=0.001,
                        early_stopping=True,
                        validation_fraction=0.15,
                        max_iter=500,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan
    rho = np.nan
    try:
        rho = spearmanr(y_true, y_pred).correlation
    except Exception:
        pass
    return rmse, mae, r2, rho


def train_baselines(descriptors, feature_matrix, labels, task_summary, n_jobs):
    id_to_index = {cid: i for i, cid in enumerate(descriptors["candidate_id"])}
    candidate_meta = descriptors[
        [
            "candidate_id",
            "preferred_name",
            "casrn",
            "dtxsid",
            "pubchem_cid",
            "inchikey",
            "has_smiles",
            "has_formula",
            "has_fluorine",
        ]
    ].copy()
    all_predict_mask = (pd.to_numeric(descriptors["has_smiles"], errors="coerce").fillna(0) > 0) | (
        pd.to_numeric(descriptors["has_formula"], errors="coerce").fillna(0) > 0
    )
    all_indices = np.where(all_predict_mask.to_numpy())[0]
    X_all_predict = feature_matrix[all_indices]
    meta_predict = candidate_meta.iloc[all_indices].reset_index(drop=True)

    metric_rows = []
    prediction_rows = []
    for _, task in task_summary.sort_values("n_candidates", ascending=False).iterrows():
        task_id = task["task_id"]
        task_labels = labels[labels["task_id"] == task_id].copy()
        task_labels = task_labels[task_labels["candidate_id"].isin(id_to_index)]
        task_labels = task_labels.drop_duplicates("candidate_id")
        n = len(task_labels)
        if n < 80:
            continue
        indices = np.array([id_to_index[cid] for cid in task_labels["candidate_id"]], dtype=int)
        y = task_labels["target_log10_mg_l"].to_numpy(dtype=float)
        test_count = max(20, int(round(0.2 * n)))
        if n < 120:
            test_count = max(15, int(round(0.25 * n)))
        test_size = min(0.35, max(0.15, test_count / float(n)))
        train_idx, test_idx, y_train, y_test = train_test_split(
            indices,
            y,
            test_size=test_size,
            random_state=RANDOM_STATE,
        )
        X_train = feature_matrix[train_idx]
        X_test = feature_matrix[test_idx]
        models = make_models(n_jobs)
        if n < 120:
            models.pop("mlp", None)
        best_name = None
        best_rmse = float("inf")
        best_model = None
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            rmse, mae, r2, rho = metrics(y_test, pred)
            metric_rows.append(
                {
                    "task_id": task_id,
                    "task_level": task["task_level"],
                    "task_label": task["task_label"],
                    "model": model_name,
                    "n_candidates": n,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "target_median_log10_mg_l": task["median_target"],
                    "rmse_log10": rmse,
                    "mae_log10": mae,
                    "r2": r2,
                    "spearman_rho": rho,
                }
            )
            if rmse < best_rmse:
                best_rmse = rmse
                best_name = model_name
                best_model = model
        best_model.fit(feature_matrix[indices], y)
        preds = best_model.predict(X_all_predict)
        pred_mg_l = np.power(10.0, preds)
        for j, meta in meta_predict.iterrows():
            prediction_rows.append(
                {
                    "task_id": task_id,
                    "task_level": task["task_level"],
                    "task_label": task["task_label"],
                    "best_model": best_name,
                    "train_candidates": n,
                    "test_rmse_log10": best_rmse,
                    "candidate_id": meta["candidate_id"],
                    "preferred_name": meta["preferred_name"],
                    "casrn": meta["casrn"],
                    "dtxsid": meta["dtxsid"],
                    "pubchem_cid": meta["pubchem_cid"],
                    "inchikey": meta["inchikey"],
                    "has_fluorine": meta["has_fluorine"],
                    "has_smiles": meta["has_smiles"],
                    "has_formula": meta["has_formula"],
                    "pred_log10_mg_l": preds[j],
                    "pred_mg_l": pred_mg_l[j],
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def build_priority(predictions, matrix):
    if predictions.empty:
        return pd.DataFrame()
    pred = predictions.copy()
    pred["endpoint"] = pred["task_label"].astype(str).str.split("|").str[0]
    pred["pred_log10_mg_l"] = pd.to_numeric(pred["pred_log10_mg_l"], errors="coerce")
    acute = pred[pred["endpoint"].isin(["LC50", "EC50"])]
    chronic = pred[pred["endpoint"].isin(["NOEC", "LOEC", "NOEL", "LOEL"])]
    acute_min = acute.groupby("candidate_id")["pred_log10_mg_l"].min().rename("pred_min_acute_log10_mg_l")
    chronic_min = chronic.groupby("candidate_id")["pred_log10_mg_l"].min().rename("pred_min_chronic_log10_mg_l")
    task_count = pred.groupby("candidate_id")["task_id"].nunique().rename("predicted_task_count")
    best_rmse = pred.groupby("candidate_id")["test_rmse_log10"].median().rename("median_task_test_rmse_log10")
    base = matrix.set_index("candidate_id")[
        [
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
    ].copy()
    out = base.join([acute_min, chronic_min, task_count, best_rmse], how="inner").reset_index()
    for col in ["wqp_records", "wqp_detected_records", "wqp_detected_fraction", "has_fluorine"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    out["toxicity_score"] = -out["pred_min_acute_log10_mg_l"].fillna(out["pred_min_chronic_log10_mg_l"])
    out["exposure_score"] = np.log10(out["wqp_detected_records"] + 1.0) + 0.2 * np.log10(out["wqp_records"] + 1.0)
    out["priority_score"] = out["toxicity_score"] + out["exposure_score"] + 0.25 * out["has_fluorine"]
    out = out.sort_values("priority_score", ascending=False)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=os.path.expanduser("~/evidence_linked_foc_project"))
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args = parser.parse_args(argv)

    base = args.base
    out_dir = os.path.join(base, "07_outputs", "stage2_modeling")
    ensure_dir(out_dir)
    print("started_utc=%s" % now())
    print("base=%s" % base)
    print("out_dir=%s" % out_dir)

    matrix, labels = load_stage1(base)
    descriptors = build_candidate_descriptors(matrix)
    usable_descriptor_ids = set(descriptors.loc[(descriptors["has_smiles"] > 0) | (descriptors["has_formula"] > 0), "candidate_id"])
    training_labels, task_summary = build_training_labels(labels, usable_descriptor_ids)
    features, numeric_cols, hash_features = build_feature_matrix(descriptors)
    metrics_df, predictions = train_baselines(descriptors, features, training_labels, task_summary, args.n_jobs)
    priority = build_priority(predictions, matrix)

    descriptors.to_csv(os.path.join(out_dir, "candidate_descriptors.tsv"), sep="\t", index=False)
    training_labels.to_csv(os.path.join(out_dir, "training_labels_long.tsv"), sep="\t", index=False)
    task_summary.to_csv(os.path.join(out_dir, "selected_task_summary.tsv"), sep="\t", index=False)
    metrics_df.to_csv(os.path.join(out_dir, "baseline_metrics.tsv"), sep="\t", index=False)
    predictions.to_csv(os.path.join(out_dir, "baseline_task_predictions.tsv"), sep="\t", index=False)
    priority.to_csv(os.path.join(out_dir, "candidate_priority_baseline.tsv"), sep="\t", index=False)

    notes = {
        "created_utc": now(),
        "candidate_rows": int(len(matrix)),
        "descriptor_rows": int(len(descriptors)),
        "usable_structure_rows": int(len(usable_descriptor_ids)),
        "raw_label_rows": int(len(labels)),
        "training_label_rows": int(len(training_labels)),
        "selected_tasks": int(task_summary["task_id"].nunique()) if len(task_summary) else 0,
        "metrics_rows": int(len(metrics_df)),
        "prediction_rows": int(len(predictions)),
        "priority_rows": int(len(priority)),
        "hash_features": int(hash_features),
        "numeric_feature_count": int(len(numeric_cols)),
        "models": ["ridge", "extra_trees", "random_forest", "mlp"],
        "feature_note": "Formula/string descriptors plus hashed SMILES character 2-4 grams; RDKit not used in this run.",
        "leakage_note": "ECOTOX availability and WQP occurrence fields are not used as toxicity model features; WQP is used only in the post-model priority score.",
    }
    with open(os.path.join(out_dir, "stage2_modeling_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(notes, handle, indent=2, sort_keys=True)
    summary_path = os.path.join(base, "01_download_logs", "processing", "stage2_modeling_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        for key in sorted(notes):
            handle.write("%s=%s\n" % (key, notes[key]))
    print(json.dumps(notes, indent=2, sort_keys=True))
    print("finished_utc=%s" % now())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
