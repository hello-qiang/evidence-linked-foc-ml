#!/usr/bin/env python3
"""Build stage-1 analysis-ready tables for model planning.

The outputs are intentionally conservative: candidate-level evidence features
and unit-normalized ECOTOX concentration labels where direct mg/L conversion is
reasonable from the reported unit alone.
"""

import argparse
import csv
import math
import os
import re
import statistics
import time
from collections import Counter, defaultdict


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def norm_text(value):
    return re.sub(r"\s+", " ", (value or "").strip())


def parse_float(value):
    value = norm_text(value)
    if not value:
        return None
    value = value.replace(",", "")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_int(value):
    number = parse_float(value)
    if number is None:
        return 0
    return int(number)


def read_tsv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def iter_tsv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            yield row


def write_tsv(path, fieldnames, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def top_items(counter, limit=5):
    return ";".join("%s:%s" % (key, value) for key, value in counter.most_common(limit))


def finite_values(values):
    return [value for value in values if value is not None and math.isfinite(value)]


def summarize_numeric(values):
    nums = sorted(finite_values(values))
    if not nums:
        return {
            "n": 0,
            "min": "",
            "median": "",
            "max": "",
            "log10_median": "",
        }
    positive = [value for value in nums if value > 0]
    log10_median = ""
    if positive:
        log10_median = "%.8g" % statistics.median([math.log10(value) for value in positive])
    return {
        "n": len(nums),
        "min": "%.8g" % nums[0],
        "median": "%.8g" % statistics.median(nums),
        "max": "%.8g" % nums[-1],
        "log10_median": log10_median,
    }


def formula_has_fluorine(formula):
    elements = re.findall(r"([A-Z][a-z]?)", formula or "")
    return "F" in elements


def smiles_has_fluorine(smiles):
    return "F" in (smiles or "")


def normalize_unit(unit):
    unit = norm_text(unit).lower()
    unit = unit.replace("\u00b5", "u").replace("\u03bc", "u")
    unit = unit.replace(" per ", "/")
    unit = re.sub(r"\s+", "", unit)
    unit = unit.replace("liter", "l").replace("litre", "l")
    return unit


def concentration_to_mg_l(value, unit):
    number = parse_float(value)
    if number is None or number <= 0:
        return None, ""
    unit_key = normalize_unit(unit)
    direct = {
        "mg/l": 1.0,
        "mg/l.": 1.0,
        "aimg/l": 1.0,
        "ai.mg/l": 1.0,
        "ug/l": 0.001,
        "ug/l.": 0.001,
        "aiug/l": 0.001,
        "ai.ug/l": 0.001,
        "ng/l": 0.000001,
        "g/l": 1000.0,
        "ppm": 1.0,
        "ppb": 0.001,
        "ppt": 0.000001,
    }
    if unit_key in direct:
        note = "direct"
        if unit_key in ("ppm", "ppb", "ppt"):
            note = "aqueous_approximation"
        return number * direct[unit_key], note
    return None, ""


def endpoint_family(endpoint):
    endpoint = norm_text(endpoint).upper()
    if endpoint in ("LC50", "LC10", "LC20"):
        return "lethal_concentration"
    if endpoint in ("EC50", "EC10", "EC20"):
        return "effect_concentration"
    if endpoint in ("IC50", "IC10", "IC20"):
        return "inhibition_concentration"
    if endpoint in ("NOEC", "NOEL"):
        return "no_effect_threshold"
    if endpoint in ("LOEC", "LOEL"):
        return "low_effect_threshold"
    if endpoint == "MATC":
        return "maximum_acceptable_toxicant_concentration"
    return "other"


def label_quality(n_records, n_species):
    if n_records >= 20 and n_species >= 3:
        return "high_support"
    if n_records >= 5:
        return "usable_sparse"
    return "screening_only"


def build_wqp_features(wqp_summary_path):
    aggregate = defaultdict(
        lambda: {
            "characteristics": set(),
            "records": 0,
            "detected_records": 0,
            "numeric_n": 0,
            "min_values": [],
            "max_values": [],
            "characteristic_counter": Counter(),
            "media": Counter(),
            "units": Counter(),
        }
    )
    for row in read_tsv(wqp_summary_path):
        cid = row.get("candidate_id", "")
        if not cid:
            continue
        item = aggregate[cid]
        records = parse_int(row.get("records", ""))
        detected = parse_int(row.get("detected_records", ""))
        item["characteristics"].add(row.get("characteristic_name", ""))
        item["records"] += records
        item["detected_records"] += detected
        item["numeric_n"] += parse_int(row.get("numeric_n", ""))
        mn = parse_float(row.get("numeric_min", ""))
        mx = parse_float(row.get("numeric_max", ""))
        if mn is not None:
            item["min_values"].append(mn)
        if mx is not None:
            item["max_values"].append(mx)
        item["characteristic_counter"][row.get("characteristic_name", "") or "NA"] += records
        for value in (row.get("top_media", "") or "").split(";"):
            if ":" in value:
                key, count = value.rsplit(":", 1)
                item["media"][key] += parse_int(count)
        for value in (row.get("top_units", "") or "").split(";"):
            if ":" in value:
                key, count = value.rsplit(":", 1)
                item["units"][key] += parse_int(count)
    rows = []
    for cid, item in aggregate.items():
        detected_fraction = ""
        if item["records"]:
            detected_fraction = "%.8g" % (float(item["detected_records"]) / float(item["records"]))
        top_characteristic = item["characteristic_counter"].most_common(1)
        top_characteristic_name = top_characteristic[0][0] if top_characteristic else ""
        top_characteristic_records = top_characteristic[0][1] if top_characteristic else 0
        rows.append(
            {
                "candidate_id": cid,
                "wqp_has_occurrence": 1 if item["records"] else 0,
                "wqp_characteristics_n": len(item["characteristics"]),
                "wqp_records": item["records"],
                "wqp_detected_records": item["detected_records"],
                "wqp_detected_fraction": detected_fraction,
                "wqp_numeric_n": item["numeric_n"],
                "wqp_numeric_min_global": "%.8g" % min(item["min_values"]) if item["min_values"] else "",
                "wqp_numeric_max_global": "%.8g" % max(item["max_values"]) if item["max_values"] else "",
                "wqp_top_characteristic": top_characteristic_name,
                "wqp_top_characteristic_records": top_characteristic_records,
                "wqp_top_media": top_items(item["media"]),
                "wqp_top_units": top_items(item["units"]),
            }
        )
    rows.sort(key=lambda row: (-int(row["wqp_records"]), row["candidate_id"]))
    return rows


def build_ecotox_tables(records_path):
    candidate = defaultdict(
        lambda: {
            "records": 0,
            "tests": set(),
            "species": set(),
            "groups": Counter(),
            "endpoints": Counter(),
            "effects": Counter(),
            "mgl_records": 0,
            "mgl_values": [],
            "label_groups": set(),
        }
    )
    labels = defaultdict(
        lambda: {
            "records": 0,
            "tests": set(),
            "species": Counter(),
            "units": Counter(),
            "values_mg_l": [],
            "conversion_notes": Counter(),
        }
    )
    selected_families = {
        "lethal_concentration",
        "effect_concentration",
        "inhibition_concentration",
        "no_effect_threshold",
        "low_effect_threshold",
        "maximum_acceptable_toxicant_concentration",
    }
    for row in iter_tsv(records_path):
        cid = row.get("candidate_id", "")
        if not cid:
            continue
        endpoint = norm_text(row.get("endpoint", "")).upper()
        effect = norm_text(row.get("effect", "")).upper()
        group = norm_text(row.get("ecotox_group", "")) or "NA"
        species = norm_text(row.get("latin_name", "")) or "NA"
        unit = norm_text(row.get("conc1_unit", ""))
        mg_l, note = concentration_to_mg_l(row.get("conc1_mean", ""), unit)
        cand = candidate[cid]
        cand["records"] += 1
        cand["tests"].add(row.get("test_id", ""))
        cand["species"].add(species)
        cand["groups"][group] += 1
        cand["endpoints"][endpoint or "NA"] += 1
        cand["effects"][effect or "NA"] += 1
        if mg_l is not None:
            cand["mgl_records"] += 1
            cand["mgl_values"].append(mg_l)
        family = endpoint_family(endpoint)
        if family not in selected_families or mg_l is None:
            continue
        key = (cid, endpoint, effect, group)
        item = labels[key]
        item["records"] += 1
        item["tests"].add(row.get("test_id", ""))
        item["species"][species] += 1
        item["units"][unit or "NA"] += 1
        item["values_mg_l"].append(mg_l)
        item["conversion_notes"][note or "direct"] += 1
        cand["label_groups"].add(key)

    label_rows = []
    for (cid, endpoint, effect, group), item in labels.items():
        stats = summarize_numeric(item["values_mg_l"])
        n_species = len(item["species"])
        label_rows.append(
            {
                "candidate_id": cid,
                "endpoint": endpoint,
                "endpoint_family": endpoint_family(endpoint),
                "effect": effect,
                "ecotox_group": group,
                "records_mg_l": stats["n"],
                "tests": len(item["tests"]),
                "species_n": n_species,
                "mg_l_min": stats["min"],
                "mg_l_median": stats["median"],
                "mg_l_max": stats["max"],
                "log10_mg_l_median": stats["log10_median"],
                "top_species": top_items(item["species"]),
                "top_original_units": top_items(item["units"]),
                "conversion_notes": top_items(item["conversion_notes"]),
                "label_quality": label_quality(stats["n"], n_species),
            }
        )
    label_rows.sort(
        key=lambda row: (
            row["candidate_id"],
            row["endpoint_family"],
            row["endpoint"],
            row["effect"],
            row["ecotox_group"],
        )
    )

    candidate_rows = []
    for cid, item in candidate.items():
        stats = summarize_numeric(item["mgl_values"])
        candidate_rows.append(
            {
                "candidate_id": cid,
                "ecotox_has_records": 1 if item["records"] else 0,
                "ecotox_records": item["records"],
                "ecotox_tests_unique": len(item["tests"]),
                "ecotox_species_unique": len(item["species"]),
                "ecotox_mgl_records": stats["n"],
                "ecotox_mgl_min": stats["min"],
                "ecotox_mgl_median": stats["median"],
                "ecotox_mgl_max": stats["max"],
                "ecotox_label_groups_mg_l": len(item["label_groups"]),
                "ecotox_top_groups": top_items(item["groups"]),
                "ecotox_top_endpoints": top_items(item["endpoints"]),
                "ecotox_top_effects": top_items(item["effects"]),
            }
        )
    candidate_rows.sort(key=lambda row: (-int(row["ecotox_records"]), row["candidate_id"]))
    return candidate_rows, label_rows


def build_candidate_matrix(candidates, wqp_features, ecotox_features):
    wqp_by_id = {row["candidate_id"]: row for row in wqp_features}
    ecotox_by_id = {row["candidate_id"]: row for row in ecotox_features}
    rows = []
    for cand in candidates:
        cid = cand.get("candidate_id", "")
        formula = cand.get("formula") or cand.get("pubchem_formula", "")
        smiles = cand.get("smiles") or cand.get("pubchem_canonical_smiles", "") or cand.get("pubchem_isomeric_smiles", "")
        pubchem_cid = cand.get("pubchem_cid") or cand.get("pubchem_cids", "")
        wqp = wqp_by_id.get(cid, {})
        eco = ecotox_by_id.get(cid, {})
        has_wqp = int(wqp.get("wqp_has_occurrence", 0) or 0)
        has_ecotox = int(eco.get("ecotox_has_records", 0) or 0)
        if has_wqp and has_ecotox:
            status = "wqp_and_ecotox"
        elif has_wqp:
            status = "wqp_only"
        elif has_ecotox:
            status = "ecotox_only"
        else:
            status = "no_stage1_match"
        completeness = 0
        for value in (pubchem_cid, cand.get("inchikey", ""), smiles, formula, cand.get("casrn", ""), cand.get("dtxsid", "")):
            if norm_text(value):
                completeness += 1
        row = {
            "candidate_id": cid,
            "preferred_name": cand.get("preferred_name", "") or cand.get("pubchem_title", ""),
            "casrn": cand.get("casrn", ""),
            "dtxsid": cand.get("dtxsid", ""),
            "pubchem_cid": pubchem_cid,
            "inchikey": cand.get("inchikey", ""),
            "smiles": smiles,
            "formula": formula,
            "pubchem_exact_mass": cand.get("pubchem_exact_mass", ""),
            "source_flags": cand.get("source_flags", ""),
            "evidence_flags": cand.get("evidence_flags", ""),
            "has_pubchem": 1 if norm_text(pubchem_cid) else 0,
            "has_inchikey": 1 if norm_text(cand.get("inchikey", "")) else 0,
            "has_smiles": 1 if norm_text(smiles) else 0,
            "has_formula": 1 if norm_text(formula) else 0,
            "has_casrn": 1 if norm_text(cand.get("casrn", "")) else 0,
            "has_dtxsid": 1 if norm_text(cand.get("dtxsid", "")) else 0,
            "has_fluorine": 1 if formula_has_fluorine(formula) or smiles_has_fluorine(smiles) else 0,
            "identifier_completeness_0_6": completeness,
            "stage1_data_status": status,
        }
        row.update(
            {
                "wqp_has_occurrence": wqp.get("wqp_has_occurrence", 0),
                "wqp_characteristics_n": wqp.get("wqp_characteristics_n", 0),
                "wqp_records": wqp.get("wqp_records", 0),
                "wqp_detected_records": wqp.get("wqp_detected_records", 0),
                "wqp_detected_fraction": wqp.get("wqp_detected_fraction", ""),
                "wqp_numeric_n": wqp.get("wqp_numeric_n", 0),
                "wqp_top_characteristic": wqp.get("wqp_top_characteristic", ""),
                "wqp_top_characteristic_records": wqp.get("wqp_top_characteristic_records", 0),
            }
        )
        row.update(
            {
                "ecotox_has_records": eco.get("ecotox_has_records", 0),
                "ecotox_records": eco.get("ecotox_records", 0),
                "ecotox_tests_unique": eco.get("ecotox_tests_unique", 0),
                "ecotox_species_unique": eco.get("ecotox_species_unique", 0),
                "ecotox_mgl_records": eco.get("ecotox_mgl_records", 0),
                "ecotox_mgl_min": eco.get("ecotox_mgl_min", ""),
                "ecotox_mgl_median": eco.get("ecotox_mgl_median", ""),
                "ecotox_mgl_max": eco.get("ecotox_mgl_max", ""),
                "ecotox_label_groups_mg_l": eco.get("ecotox_label_groups_mg_l", 0),
                "ecotox_top_groups": eco.get("ecotox_top_groups", ""),
                "ecotox_top_endpoints": eco.get("ecotox_top_endpoints", ""),
            }
        )
        rows.append(row)
    rows.sort(key=lambda row: row["candidate_id"])
    return rows


def write_manifest(out_dir, entries):
    path = os.path.join(out_dir, "stage1_data_manifest.tsv")
    fields = ["file", "rows", "description"]
    write_tsv(path, fields, entries)
    return path


def write_summary(base, data):
    path = os.path.join(base, "01_download_logs", "processing", "stage1_ml_tables_summary.txt")
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("created_utc=%s\n" % now())
        for key in sorted(data):
            handle.write("%s=%s\n" % (key, data[key]))
    return path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=os.path.expanduser("~/evidence_linked_foc_project"))
    args = parser.parse_args(argv)
    base = args.base
    out_dir = os.path.join(base, "07_outputs", "stage1_data_package")
    ensure_dir(out_dir)

    candidates_path = os.path.join(base, "03_chemical_universe", "candidate_inventory_pubchem_enriched.tsv")
    wqp_summary_path = os.path.join(base, "05_occurrence_exposure", "wqp", "wqp_candidate_occurrence_summary.tsv")
    ecotox_records_path = os.path.join(base, "04_endpoint_labels", "ecotox_candidate_records.tsv")

    print("started_utc=%s" % now())
    print("base=%s" % base)
    candidates = read_tsv(candidates_path)
    wqp_features = build_wqp_features(wqp_summary_path)
    ecotox_features, label_rows = build_ecotox_tables(ecotox_records_path)
    matrix_rows = build_candidate_matrix(candidates, wqp_features, ecotox_features)

    wqp_out = os.path.join(out_dir, "wqp_candidate_exposure_features.tsv")
    ecotox_availability_out = os.path.join(out_dir, "ecotox_candidate_label_availability.tsv")
    labels_out = os.path.join(out_dir, "ecotox_endpoint_labels_mgL.tsv")
    matrix_out = os.path.join(out_dir, "candidate_stage1_matrix.tsv")
    write_tsv(wqp_out, list(wqp_features[0].keys()) if wqp_features else ["candidate_id"], wqp_features)
    write_tsv(ecotox_availability_out, list(ecotox_features[0].keys()) if ecotox_features else ["candidate_id"], ecotox_features)
    write_tsv(labels_out, list(label_rows[0].keys()) if label_rows else ["candidate_id"], label_rows)
    write_tsv(matrix_out, list(matrix_rows[0].keys()) if matrix_rows else ["candidate_id"], matrix_rows)

    manifest_path = write_manifest(
        out_dir,
        [
            {
                "file": wqp_out,
                "rows": len(wqp_features),
                "description": "Candidate-level WQP occurrence/detection evidence aggregated from selected characteristic downloads.",
            },
            {
                "file": ecotox_availability_out,
                "rows": len(ecotox_features),
                "description": "Candidate-level ECOTOX record, endpoint, species, and mg/L-convertible label availability.",
            },
            {
                "file": labels_out,
                "rows": len(label_rows),
                "description": "Endpoint/effect/group labels from ECOTOX where concentration units can be directly converted to mg/L.",
            },
            {
                "file": matrix_out,
                "rows": len(matrix_rows),
                "description": "All candidate chemicals with identifier completeness, WQP evidence, and ECOTOX label availability features.",
            },
        ],
    )
    summary_path = write_summary(
        base,
        {
            "candidate_rows": len(candidates),
            "wqp_feature_rows": len(wqp_features),
            "ecotox_availability_rows": len(ecotox_features),
            "ecotox_mg_l_label_rows": len(label_rows),
            "candidate_matrix_rows": len(matrix_rows),
            "output_dir": out_dir,
            "manifest": manifest_path,
        },
    )
    print("summary=%s" % summary_path)
    print("output_dir=%s" % out_dir)
    print("finished_utc=%s" % now())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
