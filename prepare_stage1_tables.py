#!/usr/bin/env python3
"""Prepare stage-1 candidate-level tables for the EST project.

This script intentionally uses only the Python standard library so that the
source-table preparation step can run in minimal Python environments.
"""

import argparse
import csv
import io
import math
import os
import re
import statistics
import sys
import time
import zipfile
from collections import Counter, defaultdict


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def norm_text(value):
    return re.sub(r"\s+", " ", (value or "").strip())


def norm_key(value):
    return norm_text(value).casefold()


def norm_cas(value):
    value = norm_text(value)
    value = re.sub(r"^(CAS_RN:|CASRN:|CAS:)", "", value, flags=re.I).strip()
    if value.upper().startswith("NOCAS"):
        return ""
    digits = re.sub(r"\D+", "", value)
    return digits


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


def read_tsv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, fieldnames, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_candidates(base):
    path = os.path.join(base, "03_chemical_universe", "candidate_inventory_pubchem_enriched.tsv")
    if not os.path.exists(path):
        path = os.path.join(base, "03_chemical_universe", "candidate_inventory.tsv")
    candidates = read_tsv(path)
    by_cas = defaultdict(list)
    by_dtxsid = defaultdict(list)
    by_name = defaultdict(list)
    for row in candidates:
        cas = norm_cas(row.get("casrn", ""))
        if cas:
            by_cas[cas].append(row)
        dtxsid = norm_key(row.get("dtxsid", ""))
        if dtxsid:
            by_dtxsid[dtxsid].append(row)
        for field in ("preferred_name", "pubchem_title"):
            raw = row.get(field, "")
            for name in [part.strip() for part in raw.split(";") if part.strip()]:
                by_name[norm_key(name)].append(row)
    return candidates, by_cas, by_dtxsid, by_name


def first_candidate(rows):
    if not rows:
        return {}
    rows = sorted(rows, key=lambda r: r.get("candidate_id", ""))
    return rows[0]


def summarize_values(values):
    nums = [v for v in values if v is not None and math.isfinite(v)]
    if not nums:
        return "", "", "", ""
    nums = sorted(nums)
    return str(len(nums)), "%.8g" % nums[0], "%.8g" % statistics.median(nums), "%.8g" % nums[-1]


def build_wqp(base):
    candidates, _, _, _ = load_candidates(base)
    cand_by_id = {row.get("candidate_id", ""): row for row in candidates}
    terms_path = os.path.join(base, "05_occurrence_exposure", "wqp", "wqp_selected_candidate_terms.tsv")
    term_rows = read_tsv(terms_path)
    term_by_characteristic = defaultdict(list)
    for row in term_rows:
        term_by_characteristic[norm_key(row.get("wqp_characteristic_name", ""))].append(row)

    out_all = os.path.join(base, "05_occurrence_exposure", "wqp", "wqp_candidate_occurrence.tsv")
    out_summary = os.path.join(base, "05_occurrence_exposure", "wqp", "wqp_candidate_occurrence_summary.tsv")
    zip_dir = os.path.join(base, "05_occurrence_exposure", "wqp", "result_zips")
    ensure_dir(os.path.dirname(out_all))

    all_fields = [
        "candidate_id",
        "inchikey",
        "casrn",
        "dtxsid",
        "source_flags",
        "evidence_flags",
        "characteristic_name",
        "activity_media",
        "activity_media_subdivision",
        "sample_fraction",
        "result_detection_condition",
        "result_value",
        "result_unit",
        "measure_qualifier",
        "activity_date",
        "monitoring_location",
        "organization",
        "provider",
        "subject_taxonomic_name",
        "analytical_method",
        "dql_value",
        "dql_unit",
        "zip_file",
    ]
    summary = {}
    total_rows = 0
    matched_rows = 0
    zip_files = sorted(
        os.path.join(zip_dir, name)
        for name in os.listdir(zip_dir)
        if name.endswith(".result.zip")
    )
    with open(out_all, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields, delimiter="\t")
        writer.writeheader()
        for zip_path in zip_files:
            with zipfile.ZipFile(zip_path) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not names:
                    continue
                with zf.open(names[0]) as raw:
                    text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
                    reader = csv.DictReader(text)
                    for row in reader:
                        total_rows += 1
                        characteristic = norm_text(row.get("CharacteristicName"))
                        term_matches = term_by_characteristic.get(norm_key(characteristic), [])
                        if not term_matches:
                            continue
                        for term in term_matches:
                            cand = cand_by_id.get(term.get("candidate_id", ""), {})
                            result_value = parse_float(row.get("ResultMeasureValue"))
                            detection_condition = norm_text(row.get("ResultDetectionConditionText"))
                            detected = result_value is not None and "below" not in detection_condition.casefold()
                            key = (term.get("candidate_id", ""), characteristic)
                            item = summary.setdefault(
                                key,
                                {
                                    "candidate_id": term.get("candidate_id", ""),
                                    "inchikey": term.get("inchikey", ""),
                                    "casrn": term.get("casrn", ""),
                                    "dtxsid": term.get("dtxsid", ""),
                                    "preferred_name": cand.get("preferred_name", ""),
                                    "characteristic_name": characteristic,
                                    "records": 0,
                                    "detected_records": 0,
                                    "numeric_values": [],
                                    "media": Counter(),
                                    "units": Counter(),
                                    "providers": Counter(),
                                    "taxa": Counter(),
                                },
                            )
                            item["records"] += 1
                            if detected:
                                item["detected_records"] += 1
                            item["numeric_values"].append(result_value)
                            item["media"][norm_text(row.get("ActivityMediaName")) or "NA"] += 1
                            item["units"][norm_text(row.get("ResultMeasure/MeasureUnitCode")) or "NA"] += 1
                            item["providers"][norm_text(row.get("ProviderName")) or "NA"] += 1
                            item["taxa"][norm_text(row.get("SubjectTaxonomicName")) or "NA"] += 1
                            writer.writerow(
                                {
                                    "candidate_id": term.get("candidate_id", ""),
                                    "inchikey": term.get("inchikey", ""),
                                    "casrn": term.get("casrn", ""),
                                    "dtxsid": term.get("dtxsid", ""),
                                    "source_flags": term.get("source_flags", ""),
                                    "evidence_flags": term.get("evidence_flags", ""),
                                    "characteristic_name": characteristic,
                                    "activity_media": row.get("ActivityMediaName", ""),
                                    "activity_media_subdivision": row.get("ActivityMediaSubdivisionName", ""),
                                    "sample_fraction": row.get("ResultSampleFractionText", ""),
                                    "result_detection_condition": detection_condition,
                                    "result_value": row.get("ResultMeasureValue", ""),
                                    "result_unit": row.get("ResultMeasure/MeasureUnitCode", ""),
                                    "measure_qualifier": row.get("MeasureQualifierCode", ""),
                                    "activity_date": row.get("ActivityStartDate", ""),
                                    "monitoring_location": row.get("MonitoringLocationIdentifier", ""),
                                    "organization": row.get("OrganizationIdentifier", ""),
                                    "provider": row.get("ProviderName", ""),
                                    "subject_taxonomic_name": row.get("SubjectTaxonomicName", ""),
                                    "analytical_method": row.get("ResultAnalyticalMethod/MethodName", ""),
                                    "dql_value": row.get("DetectionQuantitationLimitMeasure/MeasureValue", ""),
                                    "dql_unit": row.get("DetectionQuantitationLimitMeasure/MeasureUnitCode", ""),
                                    "zip_file": os.path.basename(zip_path),
                                }
                            )
                            matched_rows += 1

    summary_fields = [
        "candidate_id",
        "inchikey",
        "casrn",
        "dtxsid",
        "preferred_name",
        "characteristic_name",
        "records",
        "detected_records",
        "numeric_n",
        "numeric_min",
        "numeric_median",
        "numeric_max",
        "top_media",
        "top_units",
        "top_providers",
        "top_taxa",
    ]
    rows = []
    for item in summary.values():
        n, mn, med, mx = summarize_values(item["numeric_values"])
        rows.append(
            {
                "candidate_id": item["candidate_id"],
                "inchikey": item["inchikey"],
                "casrn": item["casrn"],
                "dtxsid": item["dtxsid"],
                "preferred_name": item["preferred_name"],
                "characteristic_name": item["characteristic_name"],
                "records": item["records"],
                "detected_records": item["detected_records"],
                "numeric_n": n,
                "numeric_min": mn,
                "numeric_median": med,
                "numeric_max": mx,
                "top_media": ";".join("%s:%s" % kv for kv in item["media"].most_common(5)),
                "top_units": ";".join("%s:%s" % kv for kv in item["units"].most_common(5)),
                "top_providers": ";".join("%s:%s" % kv for kv in item["providers"].most_common(5)),
                "top_taxa": ";".join("%s:%s" % kv for kv in item["taxa"].most_common(5)),
            }
        )
    rows.sort(key=lambda r: (-int(r["records"]), r["candidate_id"], r["characteristic_name"]))
    write_tsv(out_summary, summary_fields, rows)
    write_processing_summary(
        base,
        "wqp_stage1_summary.txt",
        {
            "zip_files": len(zip_files),
            "raw_result_rows": total_rows,
            "matched_output_rows": matched_rows,
            "summary_rows": len(rows),
            "output_all": out_all,
            "output_summary": out_summary,
        },
    )


def read_pipe_from_zip(zip_path, inner_name):
    zf = zipfile.ZipFile(zip_path)
    raw = zf.open(inner_name)
    text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
    return zf, csv.DictReader(text, delimiter="|")


def build_ecotox(base):
    _, by_cas, by_dtxsid, by_name = load_candidates(base)
    zip_path = os.path.join(base, "00_source_raw", "epa_ecotox", "ecotox_ascii_03_12_2026.zip")
    out_records = os.path.join(base, "04_endpoint_labels", "ecotox_candidate_records.tsv")
    out_summary = os.path.join(base, "04_endpoint_labels", "ecotox_candidate_summary.tsv")
    ensure_dir(os.path.dirname(out_records))

    matched_chemicals = {}
    chemical_candidate_ids = defaultdict(list)
    zf, reader = read_pipe_from_zip(zip_path, "ecotox_ascii_03_12_2026/validation/chemicals.txt")
    with zf:
        for row in reader:
            cas = norm_cas(row.get("cas_number", ""))
            dtxsid = norm_key(row.get("dtxsid", ""))
            name = norm_key(row.get("chemical_name", ""))
            matches = []
            matches.extend(by_cas.get(cas, []))
            matches.extend(by_dtxsid.get(dtxsid, []))
            matches.extend(by_name.get(name, []))
            if not matches:
                continue
            candidates = {}
            for cand in matches:
                cid = cand.get("candidate_id", "")
                if cid:
                    candidates[cid] = cand
            matched_chemicals[row.get("cas_number", "")] = dict(row)
            for cid in sorted(candidates):
                chemical_candidate_ids[row.get("cas_number", "")].append(cid)

    species_by_number = {}
    zf, reader = read_pipe_from_zip(zip_path, "ecotox_ascii_03_12_2026/validation/species.txt")
    with zf:
        for row in reader:
            species_by_number[row.get("species_number", "")] = row

    test_meta = {}
    zf, reader = read_pipe_from_zip(zip_path, "ecotox_ascii_03_12_2026/tests.txt")
    with zf:
        for row in reader:
            test_cas = row.get("test_cas", "")
            if test_cas not in matched_chemicals:
                continue
            test_id = row.get("test_id", "")
            species = species_by_number.get(row.get("species_number", ""), {})
            test_meta[test_id] = {
                "test_id": test_id,
                "test_cas": test_cas,
                "reference_number": row.get("reference_number", ""),
                "species_number": row.get("species_number", ""),
                "latin_name": species.get("latin_name", ""),
                "common_name": species.get("common_name", ""),
                "ecotox_group": species.get("ecotox_group", ""),
                "study_duration_mean": row.get("study_duration_mean", ""),
                "study_duration_unit": row.get("study_duration_unit", ""),
                "exposure_duration_mean": row.get("exposure_duration_mean", ""),
                "exposure_duration_unit": row.get("exposure_duration_unit", ""),
                "study_type": row.get("study_type", ""),
                "test_type": row.get("test_type", ""),
                "exposure_type": row.get("exposure_type", ""),
                "media_type": row.get("media_type", ""),
            }

    record_fields = [
        "candidate_id",
        "ecotox_cas",
        "ecotox_chemical_name",
        "ecotox_dtxsid",
        "test_id",
        "result_id",
        "reference_number",
        "species_number",
        "latin_name",
        "common_name",
        "ecotox_group",
        "endpoint",
        "effect",
        "measurement",
        "conc1_type",
        "conc1_mean_op",
        "conc1_mean",
        "conc1_unit",
        "obs_duration_mean",
        "obs_duration_unit",
        "study_duration_mean",
        "study_duration_unit",
        "exposure_duration_mean",
        "exposure_duration_unit",
        "study_type",
        "test_type",
        "exposure_type",
        "media_type",
        "endpoint_assigned",
    ]
    summary = {}
    raw_result_rows = 0
    matched_result_rows = 0
    with open(out_records, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=record_fields, delimiter="\t")
        writer.writeheader()
        zf, reader = read_pipe_from_zip(zip_path, "ecotox_ascii_03_12_2026/results.txt")
        with zf:
            for row in reader:
                raw_result_rows += 1
                meta = test_meta.get(row.get("test_id", ""))
                if not meta:
                    continue
                chemical = matched_chemicals.get(meta["test_cas"], {})
                candidate_ids = chemical_candidate_ids.get(meta["test_cas"], [])
                for candidate_id in candidate_ids:
                    out = {
                        "candidate_id": candidate_id,
                        "ecotox_cas": meta["test_cas"],
                        "ecotox_chemical_name": chemical.get("chemical_name", ""),
                        "ecotox_dtxsid": chemical.get("dtxsid", ""),
                        "test_id": meta["test_id"],
                        "result_id": row.get("result_id", ""),
                        "reference_number": meta["reference_number"],
                        "species_number": meta["species_number"],
                        "latin_name": meta["latin_name"],
                        "common_name": meta["common_name"],
                        "ecotox_group": meta["ecotox_group"],
                        "endpoint": row.get("endpoint", ""),
                        "effect": row.get("effect", ""),
                        "measurement": row.get("measurement", ""),
                        "conc1_type": row.get("conc1_type", ""),
                        "conc1_mean_op": row.get("conc1_mean_op", ""),
                        "conc1_mean": row.get("conc1_mean", ""),
                        "conc1_unit": row.get("conc1_unit", ""),
                        "obs_duration_mean": row.get("obs_duration_mean", ""),
                        "obs_duration_unit": row.get("obs_duration_unit", ""),
                        "endpoint_assigned": row.get("endpoint_assigned", ""),
                    }
                    for key in (
                        "study_duration_mean",
                        "study_duration_unit",
                        "exposure_duration_mean",
                        "exposure_duration_unit",
                        "study_type",
                        "test_type",
                        "exposure_type",
                        "media_type",
                    ):
                        out[key] = meta.get(key, "")
                    writer.writerow(out)
                    matched_result_rows += 1
                    s_key = (candidate_id, row.get("endpoint", ""), row.get("effect", ""))
                    item = summary.setdefault(
                        s_key,
                        {
                            "candidate_id": candidate_id,
                            "endpoint": row.get("endpoint", ""),
                            "effect": row.get("effect", ""),
                            "records": 0,
                            "tests": set(),
                            "species": Counter(),
                            "values": [],
                            "units": Counter(),
                        },
                    )
                    item["records"] += 1
                    item["tests"].add(meta["test_id"])
                    item["species"][meta["latin_name"] or "NA"] += 1
                    item["values"].append(parse_float(row.get("conc1_mean")))
                    item["units"][row.get("conc1_unit", "") or "NA"] += 1

    summary_fields = [
        "candidate_id",
        "endpoint",
        "effect",
        "records",
        "tests",
        "numeric_n",
        "numeric_min",
        "numeric_median",
        "numeric_max",
        "top_species",
        "top_units",
    ]
    rows = []
    for item in summary.values():
        n, mn, med, mx = summarize_values(item["values"])
        rows.append(
            {
                "candidate_id": item["candidate_id"],
                "endpoint": item["endpoint"],
                "effect": item["effect"],
                "records": item["records"],
                "tests": len(item["tests"]),
                "numeric_n": n,
                "numeric_min": mn,
                "numeric_median": med,
                "numeric_max": mx,
                "top_species": ";".join("%s:%s" % kv for kv in item["species"].most_common(5)),
                "top_units": ";".join("%s:%s" % kv for kv in item["units"].most_common(5)),
            }
        )
    rows.sort(key=lambda r: (-int(r["records"]), r["candidate_id"], r["endpoint"]))
    write_tsv(out_summary, summary_fields, rows)
    write_processing_summary(
        base,
        "ecotox_stage1_summary.txt",
        {
            "matched_ecotox_chemicals": len(matched_chemicals),
            "matched_ecotox_tests": len(test_meta),
            "raw_result_rows_scanned": raw_result_rows,
            "matched_output_rows": matched_result_rows,
            "summary_rows": len(rows),
            "output_records": out_records,
            "output_summary": out_summary,
        },
    )


def write_processing_summary(base, name, data):
    out = os.path.join(base, "01_download_logs", "processing", name)
    ensure_dir(os.path.dirname(out))
    with open(out, "w", encoding="utf-8") as handle:
        handle.write("created_utc=%s\n" % now())
        for key in sorted(data):
            handle.write("%s=%s\n" % (key, data[key]))
    print("wrote_summary=%s" % out)
    for key in sorted(data):
        print("%s=%s" % (key, data[key]))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=os.path.expanduser("~/evidence_linked_foc_project"))
    parser.add_argument("--task", choices=["wqp", "ecotox"], required=True)
    args = parser.parse_args(argv)
    print("task=%s" % args.task)
    print("base=%s" % args.base)
    print("started_utc=%s" % now())
    if args.task == "wqp":
        build_wqp(args.base)
    elif args.task == "ecotox":
        build_ecotox(args.base)
    print("finished_utc=%s" % now())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
