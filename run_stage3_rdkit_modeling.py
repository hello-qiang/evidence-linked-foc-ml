#!/usr/bin/env python3
"""Stage-3 RDKit descriptors, Morgan fingerprints, and scaffold split models."""

import argparse
import json
import math
import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator


RDLogger.DisableLog("rdApp.*")
RANDOM_STATE = 20260607
FP_SIZE = 1024


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def finite_float(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else np.nan
    except Exception:
        return np.nan


def chemical_family(mol, smiles):
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    f_count = atoms.count("F")
    clbr_count = atoms.count("Cl") + atoms.count("Br") + atoms.count("I")
    ring_count = rdMolDescriptors.CalcNumRings(mol)
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    hetero = sum(1 for atom in atoms if atom not in ("C", "H", "F", "Cl", "Br", "I"))
    carbon = atoms.count("C")
    smiles = smiles or ""
    if f_count >= 6 and carbon >= 4:
        return "polyfluorinated"
    if f_count > 0 and aromatic_atoms:
        return "fluorinated_aromatic"
    if f_count > 0:
        return "fluorinated_nonaromatic"
    if clbr_count > 0 and aromatic_atoms:
        return "chlorobromo_aromatic"
    if clbr_count > 0:
        return "chlorobromo_nonaromatic"
    if aromatic_atoms and hetero:
        return "heteroaromatic"
    if aromatic_atoms:
        return "aromatic_hydrocarbon_like"
    if hetero:
        return "heteroaliphatic"
    if ring_count:
        return "carbocyclic"
    return "aliphatic"


def atom_count(mol, symbol):
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == symbol)


def safe_descriptor(func, mol):
    try:
        value = func(mol)
        return finite_float(value)
    except Exception:
        return np.nan


def murcko_scaffold(mol):
    try:
        value = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return value or ""
    except Exception:
        return ""


def descriptor_row(candidate_id, smiles, metadata):
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
    row = OrderedDict()
    row["candidate_id"] = candidate_id
    row["smiles"] = smiles
    row["rdkit_mol_ok"] = 1 if mol is not None else 0
    for key in [
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
    ]:
        row[key] = metadata.get(key, "")
    if mol is None:
        row["chemical_family"] = "invalid_or_missing_smiles"
        row["murcko_scaffold"] = ""
        row["split_group"] = "invalid_or_missing_smiles"
        return row, None
    row["canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    row["isomeric_canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    scaffold = murcko_scaffold(mol)
    family = chemical_family(mol, smiles)
    heavy = mol.GetNumHeavyAtoms()
    ring_count = safe_descriptor(rdMolDescriptors.CalcNumRings, mol)
    halogens = atom_count(mol, "F") + atom_count(mol, "Cl") + atom_count(mol, "Br") + atom_count(mol, "I")
    row["chemical_family"] = family
    row["murcko_scaffold"] = scaffold
    heavy_bin = int(min(9, max(0, heavy // 10)))
    halogen_bin = int(min(9, max(0, halogens)))
    if scaffold:
        row["split_group"] = "scaffold:%s" % scaffold
    else:
        row["split_group"] = "acyclic:%s:H%s:X%s" % (family, heavy_bin, halogen_bin)
    descriptors = OrderedDict(
        [
            ("rdkit_MolWt", Descriptors.MolWt),
            ("rdkit_ExactMolWt", Descriptors.ExactMolWt),
            ("rdkit_MolLogP", Crippen.MolLogP),
            ("rdkit_MolMR", Crippen.MolMR),
            ("rdkit_TPSA", rdMolDescriptors.CalcTPSA),
            ("rdkit_LabuteASA", rdMolDescriptors.CalcLabuteASA),
            ("rdkit_BertzCT", Descriptors.BertzCT),
            ("rdkit_HeavyAtomCount", Descriptors.HeavyAtomCount),
            ("rdkit_NumValenceElectrons", Descriptors.NumValenceElectrons),
            ("rdkit_NumHAcceptors", Lipinski.NumHAcceptors),
            ("rdkit_NumHDonors", Lipinski.NumHDonors),
            ("rdkit_NumRotatableBonds", Lipinski.NumRotatableBonds),
            ("rdkit_RingCount", rdMolDescriptors.CalcNumRings),
            ("rdkit_NumAromaticRings", Lipinski.NumAromaticRings),
            ("rdkit_NumAliphaticRings", Lipinski.NumAliphaticRings),
            ("rdkit_NumSaturatedRings", Lipinski.NumSaturatedRings),
            ("rdkit_FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
            ("rdkit_FormalCharge", Chem.GetFormalCharge),
            ("rdkit_QED", QED.qed),
        ]
    )
    for name, func in descriptors.items():
        row[name] = safe_descriptor(func, mol)
    row["rdkit_F_count"] = atom_count(mol, "F")
    row["rdkit_Cl_count"] = atom_count(mol, "Cl")
    row["rdkit_Br_count"] = atom_count(mol, "Br")
    row["rdkit_I_count"] = atom_count(mol, "I")
    row["rdkit_N_count"] = atom_count(mol, "N")
    row["rdkit_O_count"] = atom_count(mol, "O")
    row["rdkit_S_count"] = atom_count(mol, "S")
    row["rdkit_P_count"] = atom_count(mol, "P")
    row["rdkit_halogen_count"] = halogens
    row["rdkit_halogen_fraction_heavy"] = halogens / heavy if heavy else 0.0
    row["rdkit_f_fraction_heavy"] = row["rdkit_F_count"] / heavy if heavy else 0.0
    return row, mol


def build_rdkit_features(matrix):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)
    rows = []
    fps = []
    valid_ids = []
    for _, item in matrix.iterrows():
        cid = item["candidate_id"]
        smiles = item.get("smiles", "")
        row, mol = descriptor_row(cid, smiles, item)
        rows.append(row)
        if mol is not None:
            fp = generator.GetFingerprint(mol)
            arr = np.zeros((FP_SIZE,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            valid_ids.append(cid)
    descriptors = pd.DataFrame(rows)
    fp_matrix = np.vstack(fps).astype(np.float32) if fps else np.zeros((0, FP_SIZE), dtype=np.float32)
    return descriptors, valid_ids, fp_matrix


def feature_matrix(descriptors, valid_ids, fp_matrix):
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
        if col not in exclude and col.startswith(("rdkit_", "has_fluorine"))
    ]
    numeric = valid[numeric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    X = np.hstack([numeric, fp_matrix]).astype(np.float32)
    return valid, X, numeric_cols


def metrics(y_true, y_pred):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan
    try:
        rho = spearmanr(y_true, y_pred).correlation
    except Exception:
        rho = np.nan
    return rmse, mae, r2, rho


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
                        n_estimators=500,
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
                        n_estimators=400,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        random_state=RANDOM_STATE,
                        n_jobs=n_jobs,
                    ),
                ),
            ]
        ),
    }


def domain_stats(task_labels, valid):
    joined = task_labels.merge(
        valid[["candidate_id", "rdkit_MolWt", "rdkit_HeavyAtomCount", "rdkit_halogen_count", "rdkit_MolLogP", "rdkit_TPSA"]],
        on="candidate_id",
        how="left",
    )
    stats = {}
    for col in ["rdkit_MolWt", "rdkit_HeavyAtomCount", "rdkit_halogen_count", "rdkit_MolLogP", "rdkit_TPSA"]:
        values = pd.to_numeric(joined[col], errors="coerce").dropna()
        if len(values):
            q01, q99 = values.quantile([0.01, 0.99])
            q05, q95 = values.quantile([0.05, 0.95])
            spread = max(float(q95 - q05), 1.0)
            stats[col] = (max(float(q01 - 0.2 * spread), 0.0), float(q99 + 0.2 * spread))
        else:
            stats[col] = (np.nan, np.nan)
    target = pd.to_numeric(task_labels["target_log10_mg_l"], errors="coerce").dropna()
    if len(target):
        stats["target_q01_q99"] = (float(target.quantile(0.01)), float(target.quantile(0.99)))
    else:
        stats["target_q01_q99"] = (np.nan, np.nan)
    return stats


def apply_domain(valid, stats):
    mask = np.ones(len(valid), dtype=bool)
    for col, (lower, upper) in stats.items():
        if col == "target_q01_q99":
            continue
        values = pd.to_numeric(valid[col], errors="coerce")
        mask &= values.notna().to_numpy()
        mask &= (values >= lower).to_numpy()
        mask &= (values <= upper).to_numpy()
    return mask.astype(int)


def evaluate_and_predict(valid, X, labels, n_jobs):
    id_to_index = {cid: i for i, cid in enumerate(valid["candidate_id"])}
    metrics_rows = []
    prediction_parts = []
    domain_rows = []
    valid_ids = set(id_to_index)
    labels = labels[labels["candidate_id"].isin(valid_ids)].copy()
    task_summary = (
        labels.groupby(["task_id", "task_level", "task_label"])
        .agg(n_candidates=("candidate_id", "nunique"), rows=("candidate_id", "size"), median_target=("target_log10_mg_l", "median"))
        .reset_index()
        .sort_values("n_candidates", ascending=False)
    )
    for _, task in task_summary.iterrows():
        task_id = task["task_id"]
        task_labels = labels[labels["task_id"] == task_id].drop_duplicates("candidate_id").copy()
        n = len(task_labels)
        if n < 80:
            continue
        indices = np.array([id_to_index[cid] for cid in task_labels["candidate_id"]], dtype=int)
        y = task_labels["target_log10_mg_l"].to_numpy(dtype=float)
        groups = valid.loc[indices, "split_group"].fillna("missing_group").astype(str).to_numpy()
        unique_groups = len(set(groups))
        split_specs = []
        train_idx, test_idx, y_train, y_test = train_test_split(
            indices, y, test_size=max(0.2, min(0.3, 25 / float(max(n, 1)))), random_state=RANDOM_STATE
        )
        split_specs.append(("random", train_idx, test_idx, y_train, y_test))
        if unique_groups >= 8:
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
            local_train, local_test = next(splitter.split(indices, y, groups=groups))
            if len(local_test) >= 15 and len(local_train) >= 50:
                split_specs.append(("scaffold_group", indices[local_train], indices[local_test], y[local_train], y[local_test]))
        best_model_name = None
        best_model = None
        best_scaffold_rmse = float("inf")
        best_random_rmse = float("inf")
        for split_name, train_abs, test_abs, y_train, y_test in split_specs:
            X_train = X[train_abs]
            X_test = X[test_abs]
            for model_name, model in make_models(n_jobs).items():
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                rmse, mae, r2, rho = metrics(y_test, pred)
                metrics_rows.append(
                    {
                        "task_id": task_id,
                        "task_level": task["task_level"],
                        "task_label": task["task_label"],
                        "split": split_name,
                        "model": model_name,
                        "n_candidates": n,
                        "n_groups": unique_groups,
                        "n_train": len(train_abs),
                        "n_test": len(test_abs),
                        "rmse_log10": rmse,
                        "mae_log10": mae,
                        "r2": r2,
                        "spearman_rho": rho,
                    }
                )
                if split_name == "scaffold_group":
                    if rmse < best_scaffold_rmse:
                        best_scaffold_rmse = rmse
                        best_model_name = model_name
                elif split_name == "random" and rmse < best_random_rmse:
                    best_random_rmse = rmse
                    if best_model_name is None:
                        best_model_name = model_name
        best_model_name = best_model_name or "extra_trees"
        best_model = make_models(n_jobs)[best_model_name]
        best_model.fit(X[indices], y)
        stats = domain_stats(task_labels, valid)
        ad_pass = apply_domain(valid, stats)
        lo, hi = stats["target_q01_q99"]
        preds = best_model.predict(X)
        screened = np.clip(preds, lo, hi)
        pred_part = valid[
            ["candidate_id", "preferred_name", "casrn", "dtxsid", "pubchem_cid", "inchikey", "has_fluorine"]
        ].copy()
        pred_part["task_id"] = task_id
        pred_part["task_level"] = task["task_level"]
        pred_part["task_label"] = task["task_label"]
        pred_part["best_model_by_scaffold_or_random"] = best_model_name
        pred_part["train_candidates"] = n
        pred_part["best_scaffold_rmse_log10"] = best_scaffold_rmse if math.isfinite(best_scaffold_rmse) else ""
        pred_part["best_random_rmse_log10"] = best_random_rmse
        pred_part["pred_log10_mg_l"] = preds
        pred_part["pred_log10_mg_l_screened"] = screened
        pred_part["pred_mg_l_screened"] = np.power(10.0, screened)
        pred_part["rdkit_ad_pass"] = ad_pass
        prediction_parts.append(pred_part)
        domain_row = {
            "task_id": task_id,
            "task_level": task["task_level"],
            "task_label": task["task_label"],
            "n_candidates": n,
            "n_groups": unique_groups,
            "target_q01": lo,
            "target_q99": hi,
        }
        for col, value in stats.items():
            if col == "target_q01_q99":
                continue
            domain_row[col + "_lower"] = value[0]
            domain_row[col + "_upper"] = value[1]
        domain_rows.append(domain_row)
    metrics_df = pd.DataFrame(metrics_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    domains = pd.DataFrame(domain_rows)
    return task_summary, metrics_df, predictions, domains


def build_priority(predictions, matrix):
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    pred = predictions.copy()
    pred["endpoint"] = pred["task_label"].astype(str).str.split("|").str[0]
    acute = pred[(pred["endpoint"].isin(["LC50", "EC50"])) & (pd.to_numeric(pred["rdkit_ad_pass"], errors="coerce") == 1)]
    best = (
        acute.sort_values(["candidate_id", "pred_log10_mg_l_screened"])
        .groupby("candidate_id")
        .head(1)
        .rename(columns={"pred_log10_mg_l_screened": "rdkit_pred_min_acute_log10_mg_l"})
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
    out = base.merge(
        best[
            [
                "candidate_id",
                "task_id",
                "task_label",
                "best_model_by_scaffold_or_random",
                "best_scaffold_rmse_log10",
                "best_random_rmse_log10",
                "rdkit_pred_min_acute_log10_mg_l",
                "pred_mg_l_screened",
                "rdkit_ad_pass",
            ]
        ],
        on="candidate_id",
        how="inner",
    )
    for col in ["has_fluorine", "wqp_records", "wqp_detected_records", "wqp_detected_fraction"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    rmse = pd.to_numeric(out["best_scaffold_rmse_log10"].replace("", np.nan), errors="coerce")
    rmse = rmse.fillna(pd.to_numeric(out["best_random_rmse_log10"], errors="coerce")).fillna(1.5)
    out["exposure_score"] = np.log10(out["wqp_detected_records"] + 1.0) + 0.2 * np.log10(out["wqp_records"] + 1.0)
    out["toxicity_score"] = -pd.to_numeric(out["rdkit_pred_min_acute_log10_mg_l"], errors="coerce")
    out["rdkit_priority_score"] = out["toxicity_score"] + out["exposure_score"] + 0.25 * out["has_fluorine"] - 0.25 * rmse
    all_priority = out.sort_values("rdkit_priority_score", ascending=False)
    exposed_priority = all_priority[all_priority["wqp_records"] > 0].copy()
    return exposed_priority, all_priority


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=os.path.expanduser("~/evidence_linked_foc_project"))
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args = parser.parse_args(argv)
    base = args.base
    stage1 = os.path.join(base, "07_outputs", "stage1_data_package")
    stage2 = os.path.join(base, "07_outputs", "stage2_modeling")
    out_dir = os.path.join(base, "07_outputs", "stage3_rdkit_modeling")
    ensure_dir(out_dir)
    print("started_utc=%s" % now())
    print("base=%s" % base)
    matrix = pd.read_csv(os.path.join(stage1, "candidate_stage1_matrix.tsv"), sep="\t", dtype=str).fillna("")
    labels = pd.read_csv(os.path.join(stage2, "training_labels_long.tsv"), sep="\t")
    labels["target_log10_mg_l"] = pd.to_numeric(labels["target_log10_mg_l"], errors="coerce")
    labels = labels[np.isfinite(labels["target_log10_mg_l"])].copy()
    descriptors, valid_ids, fp_matrix = build_rdkit_features(matrix)
    valid, X, numeric_cols = feature_matrix(descriptors, valid_ids, fp_matrix)
    task_summary, metrics_df, predictions, domains = evaluate_and_predict(valid, X, labels, args.n_jobs)
    exposed_priority, all_priority = build_priority(predictions, matrix)

    descriptors.to_csv(os.path.join(out_dir, "candidate_rdkit_descriptors.tsv"), sep="\t", index=False)
    valid[["candidate_id", "chemical_family", "murcko_scaffold", "split_group"]].to_csv(
        os.path.join(out_dir, "candidate_scaffold_groups.tsv"), sep="\t", index=False
    )
    np.save(os.path.join(out_dir, "candidate_morgan1024_valid_ids.npy"), np.array(valid_ids, dtype=str))
    np.save(os.path.join(out_dir, "candidate_morgan1024.npy"), fp_matrix)
    task_summary.to_csv(os.path.join(out_dir, "rdkit_task_summary.tsv"), sep="\t", index=False)
    metrics_df.to_csv(os.path.join(out_dir, "rdkit_model_metrics.tsv"), sep="\t", index=False)
    predictions.to_csv(os.path.join(out_dir, "rdkit_task_predictions.tsv"), sep="\t", index=False)
    domains.to_csv(os.path.join(out_dir, "rdkit_applicability_domain_by_task.tsv"), sep="\t", index=False)
    exposed_priority.to_csv(os.path.join(out_dir, "candidate_priority_exposed_rdkit.tsv"), sep="\t", index=False)
    all_priority.to_csv(os.path.join(out_dir, "candidate_priority_all_rdkit.tsv"), sep="\t", index=False)
    summary = {
        "created_utc": now(),
        "candidate_rows": int(len(matrix)),
        "rdkit_valid_molecules": int(len(valid_ids)),
        "fingerprint_bits": FP_SIZE,
        "rdkit_numeric_descriptors": int(len(numeric_cols)),
        "training_label_rows": int(len(labels)),
        "task_rows": int(len(task_summary)),
        "metrics_rows": int(len(metrics_df)),
        "prediction_rows": int(len(predictions)),
        "exposed_priority_rows": int(len(exposed_priority)),
        "all_priority_rows": int(len(all_priority)),
        "rdkit_dependency": "rdkit 2024.3.2 installed under project pydeps with --no-deps",
        "split_note": "Metrics include random and scaffold_group splits; scaffold_group uses Murcko scaffold when present and coarse acyclic chemical-family bins otherwise.",
    }
    with open(os.path.join(out_dir, "stage3_rdkit_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with open(os.path.join(base, "01_download_logs", "processing", "stage3_rdkit_summary.txt"), "w", encoding="utf-8") as handle:
        for key in sorted(summary):
            handle.write("%s=%s\n" % (key, summary[key]))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("finished_utc=%s" % now())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
