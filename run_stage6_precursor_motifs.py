#!/usr/bin/env python3
"""Stage-6 structural precursor motif screening for fluorinated candidates."""

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


SMARTS = {
    "cf3": "[CX4](F)(F)F",
    "sulfonamide": "S(=O)(=O)N",
    "sulfonate_or_sulfate": "S(=O)(=O)O",
    "amide": "C(=O)N",
    "ester": "C(=O)O[#6]",
    "ether": "[#6]-O-[#6]",
    "azole_like": "[nH0,nH1,o,s]1~*~*~*~*1",
}


def count_atoms(mol, symbol):
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == symbol)


def count_smarts(mol, query):
    patt = Chem.MolFromSmarts(query)
    if patt is None:
        return 0
    return len(mol.GetSubstructMatches(patt, uniquify=True))


def max_pf_carbon_component(mol):
    pf_nodes = set()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C" or atom.GetIsAromatic():
            continue
        f_neighbors = sum(1 for nbr in atom.GetNeighbors() if nbr.GetSymbol() == "F")
        carbon_neighbors = sum(1 for nbr in atom.GetNeighbors() if nbr.GetSymbol() == "C")
        hetero_neighbors = sum(1 for nbr in atom.GetNeighbors() if nbr.GetSymbol() not in ("C", "H", "F"))
        if f_neighbors >= 2 or (f_neighbors >= 1 and carbon_neighbors >= 1 and hetero_neighbors == 0):
            pf_nodes.add(atom.GetIdx())
    seen = set()
    best = 0
    for node in pf_nodes:
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            atom = mol.GetAtomWithIdx(cur)
            for nbr in atom.GetNeighbors():
                ni = nbr.GetIdx()
                if ni in pf_nodes and ni not in seen:
                    seen.add(ni)
                    stack.append(ni)
        best = max(best, size)
    return best


def motif_row(row):
    smiles = row.get("smiles", "")
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
    out = {
        "candidate_id": row.get("candidate_id", ""),
        "preferred_name": row.get("preferred_name", ""),
        "casrn": row.get("casrn", ""),
        "dtxsid": row.get("dtxsid", ""),
        "pubchem_cid": row.get("pubchem_cid", ""),
        "inchikey": row.get("inchikey", ""),
        "chemical_family": row.get("chemical_family", ""),
        "smiles": smiles,
        "motif_parse_ok": int(mol is not None),
    }
    if mol is None:
        for key in [
            "f_count",
            "cf3_count",
            "max_pf_carbon_component",
            "sulfonamide_count",
            "sulfonate_or_sulfate_count",
            "amide_count",
            "ester_count",
            "ether_count",
            "azole_like_count",
            "fluorinated_aromatic_bonds",
            "tfa_plausible_motif",
            "pfaa_chain_motif",
            "pfas_precursor_motif",
            "labile_linkage_motif",
            "precursor_motif_score",
        ]:
            out[key] = 0
        return out
    f_count = count_atoms(mol, "F")
    aromatic_cf = 0
    for bond in mol.GetBonds():
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        if (a.GetIsAromatic() and b.GetSymbol() == "F") or (b.GetIsAromatic() and a.GetSymbol() == "F"):
            aromatic_cf += 1
    cf3_count = count_smarts(mol, SMARTS["cf3"])
    pf_component = max_pf_carbon_component(mol)
    sulfonamide = count_smarts(mol, SMARTS["sulfonamide"])
    sulfonate = count_smarts(mol, SMARTS["sulfonate_or_sulfate"])
    amide = count_smarts(mol, SMARTS["amide"])
    ester = count_smarts(mol, SMARTS["ester"])
    ether = count_smarts(mol, SMARTS["ether"])
    azole = count_smarts(mol, SMARTS["azole_like"])
    labile = int((amide + ester + ether + sulfonamide + sulfonate) > 0)
    tfa_plausible = int(cf3_count > 0)
    pfaa_chain = int(pf_component >= 2 or f_count >= 6)
    pfas_precursor = int(pf_component >= 3 or (f_count >= 6 and (sulfonamide + sulfonate) > 0))
    score = (
        min(cf3_count, 3) * 0.75
        + min(pf_component, 6) * 0.45
        + min(f_count, 12) * 0.05
        + 0.50 * sulfonamide
        + 0.35 * sulfonate
        + 0.20 * labile
        + 0.15 * min(aromatic_cf, 4)
    )
    out.update(
        {
            "f_count": f_count,
            "cf3_count": cf3_count,
            "max_pf_carbon_component": pf_component,
            "sulfonamide_count": sulfonamide,
            "sulfonate_or_sulfate_count": sulfonate,
            "amide_count": amide,
            "ester_count": ester,
            "ether_count": ether,
            "azole_like_count": azole,
            "fluorinated_aromatic_bonds": aromatic_cf,
            "tfa_plausible_motif": tfa_plausible,
            "pfaa_chain_motif": pfaa_chain,
            "pfas_precursor_motif": pfas_precursor,
            "labile_linkage_motif": labile,
            "precursor_motif_score": score,
        }
    )
    return out


def motif_screen(base):
    descriptors = pd.read_csv(
        os.path.join(base, "07_outputs", "stage3_rdkit_modeling", "candidate_rdkit_descriptors.tsv"),
        sep="\t",
        dtype=str,
    ).fillna("")
    rows = [motif_row(row) for _, row in descriptors.iterrows()]
    return pd.DataFrame(rows)


def integrate_priority(base, motifs):
    priority_path = os.path.join(base, "07_outputs", "stage5_family_transfer", "stage5_transfer_aware_priority.tsv")
    if not os.path.exists(priority_path):
        priority_path = os.path.join(base, "07_outputs", "stage4_robustness_uncertainty", "stage4_priority_uncertainty.tsv")
    priority = pd.read_csv(priority_path, sep="\t", dtype=str).fillna("")
    for col in ["stage4_priority_score", "pred_log10_sd", "uncertainty_band_log10", "wqp_detected_records", "wqp_records"]:
        if col in priority.columns:
            priority[col] = pd.to_numeric(priority[col], errors="coerce")
    keep = [
        "candidate_id",
        "chemical_family",
        "f_count",
        "cf3_count",
        "max_pf_carbon_component",
        "sulfonamide_count",
        "sulfonate_or_sulfate_count",
        "tfa_plausible_motif",
        "pfaa_chain_motif",
        "pfas_precursor_motif",
        "labile_linkage_motif",
        "precursor_motif_score",
    ]
    out = priority.merge(motifs[keep], on="candidate_id", how="left", suffixes=("", "_motif"))
    out["precursor_integrated_score"] = out["stage4_priority_score"].fillna(0) + 0.55 * out["precursor_motif_score"].fillna(0)
    def tier(row):
        action = row.get("transfer_aware_action_tier", "")
        precursor = row.get("precursor_motif_score", 0)
        if precursor >= 3.0 and str(action).startswith("Tier 1"):
            return "Precursor Tier A: high-confidence priority precursor"
        if precursor >= 2.0 and str(action).startswith(("Tier 1", "Tier 2")):
            return "Precursor Tier B: strong precursor evidence"
        if precursor >= 1.0:
            return "Precursor Tier C: motif-supported follow-up"
        return "Precursor Tier D: weak structural precursor evidence"
    out["precursor_action_tier"] = out.apply(tier, axis=1)
    return out.sort_values(["precursor_action_tier", "precursor_integrated_score"], ascending=[True, False])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=os.path.expanduser("~/evidence_linked_foc_project"))
    args = parser.parse_args(argv)
    base = args.base
    out_dir = os.path.join(base, "07_outputs", "stage6_precursor_motifs")
    ensure_dir(out_dir)
    print("started_utc=%s" % now())
    print("base=%s" % base)
    motifs = motif_screen(base)
    priority = integrate_priority(base, motifs)
    family_summary = (
        motifs.groupby("chemical_family")
        .agg(
            candidates=("candidate_id", "size"),
            median_f_count=("f_count", "median"),
            tfa_plausible=("tfa_plausible_motif", "sum"),
            pfaa_chain=("pfaa_chain_motif", "sum"),
            pfas_precursor=("pfas_precursor_motif", "sum"),
            median_precursor_score=("precursor_motif_score", "median"),
        )
        .reset_index()
        .sort_values("median_precursor_score", ascending=False)
    )
    motifs.to_csv(os.path.join(out_dir, "stage6_precursor_motif_flags.tsv"), sep="\t", index=False)
    family_summary.to_csv(os.path.join(out_dir, "stage6_family_motif_summary.tsv"), sep="\t", index=False)
    priority.to_csv(os.path.join(out_dir, "stage6_precursor_priority.tsv"), sep="\t", index=False)
    summary = {
        "created_utc": now(),
        "motif_rows": int(len(motifs)),
        "priority_rows": int(len(priority)),
        "tfa_plausible_candidates": int(motifs["tfa_plausible_motif"].sum()),
        "pfaa_chain_candidates": int(motifs["pfaa_chain_motif"].sum()),
        "pfas_precursor_candidates": int(motifs["pfas_precursor_motif"].sum()),
        "top_precursor_priority": priority[["preferred_name", "precursor_integrated_score"]].head(10).to_dict("records"),
        "note": "Heuristic structural motif screen for TFA/PFAA/PFAS precursor potential integrated with Stage5 transfer-aware priority.",
    }
    with open(os.path.join(out_dir, "stage6_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with open(os.path.join(base, "01_download_logs", "processing", "stage6_precursor_motif_summary.txt"), "w", encoding="utf-8") as handle:
        for key in sorted(summary):
            handle.write("%s=%s\n" % (key, summary[key]))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("finished_utc=%s" % now())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
