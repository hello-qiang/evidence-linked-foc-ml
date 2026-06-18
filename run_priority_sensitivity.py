#!/usr/bin/env python3
"""Build fluorinated-main priority queues and ranking sensitivity tables.

This post-processing layer does not retrain toxicity models. It audits how the
exposed priority queue changes when occurrence weighting, fluorination terms,
uncertainty, and confidence filters are varied.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TOP_K = (10, 20, 50)


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_rank(df, score_col, rank_col):
    out = df.sort_values(score_col, ascending=False).copy()
    out[rank_col] = np.arange(1, len(out) + 1)
    return out


def rank_column(df, score_col):
    return df[score_col].rank(method="first", ascending=False).astype(int)


def build_scores(priority):
    df = priority.copy()
    df = numeric(
        df,
        [
            "has_fluorine",
            "wqp_records",
            "wqp_detected_records",
            "wqp_detected_fraction",
            "toxicity_score_mean",
            "exposure_score",
            "pred_log10_sd",
            "stage4_priority_score",
            "precursor_motif_score",
            "precursor_integrated_score",
        ],
    )
    df["fraction_exposure_score"] = np.log10(1.0 + 100.0 * df["wqp_detected_fraction"].fillna(0))
    df["occurrence_score"] = df["exposure_score"]
    df["score_stage4_recomputed"] = (
        df["toxicity_score_mean"]
        + df["occurrence_score"]
        + 0.25 * df["has_fluorine"]
        - 0.50 * df["pred_log10_sd"]
    )
    df["score_no_fluorination_term"] = df["score_stage4_recomputed"] - 0.25 * df["has_fluorine"]
    df["score_fraction_only_occurrence"] = (
        df["toxicity_score_mean"]
        + df["fraction_exposure_score"]
        + 0.25 * df["has_fluorine"]
        - 0.50 * df["pred_log10_sd"]
    )
    df["score_half_occurrence_weight"] = (
        df["toxicity_score_mean"]
        + 0.50 * df["occurrence_score"]
        + 0.25 * df["has_fluorine"]
        - 0.50 * df["pred_log10_sd"]
    )
    df["score_hazard_uncertainty_only"] = (
        df["toxicity_score_mean"]
        + 0.25 * df["has_fluorine"]
        - 0.50 * df["pred_log10_sd"]
    )
    df["score_precursor_integrated"] = df.get("precursor_integrated_score", np.nan)
    return df


def topk_overlap(df, variants):
    rows = []
    for k in TOP_K:
        top_sets = {}
        for name, col, subset in variants:
            sub = df.copy()
            if subset is not None:
                sub = sub[subset(sub)].copy()
            sub = sub.sort_values(col, ascending=False).head(k)
            top_sets[name] = set(sub["candidate_id"].astype(str))
        names = list(top_sets)
        for i, a in enumerate(names):
            for b in names[i:]:
                A, B = top_sets[a], top_sets[b]
                union = A | B
                rows.append(
                    {
                        "top_k": k,
                        "variant_a": a,
                        "variant_b": b,
                        "overlap_n": len(A & B),
                        "jaccard": len(A & B) / len(union) if union else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def variant_summary(df, variants):
    rows = []
    for name, col, subset in variants:
        sub = df.copy()
        if subset is not None:
            sub = sub[subset(sub)].copy()
        sub = sub.sort_values(col, ascending=False)
        rows.append(
            {
                "variant": name,
                "score_column": col,
                "n_candidates": len(sub),
                "top10_candidates": "; ".join(sub["preferred_name"].astype(str).head(10)),
                "top20_fluorinated_count": int(sub.head(20)["has_fluorine"].fillna(0).astype(int).sum()),
                "median_detected_fraction_top20": sub.head(20)["wqp_detected_fraction"].median(),
                "median_prediction_sd_top20": sub.head(20)["pred_log10_sd"].median(),
            }
        )
    return pd.DataFrame(rows)


def endpoint_reliability(metrics):
    df = metrics.copy()
    df = numeric(df, ["rmse_log10_scaffold_group", "spearman_rho_scaffold_group", "r2_scaffold_group"])

    def tier(row):
        rho = row["spearman_rho_scaffold_group"]
        rmse = row["rmse_log10_scaffold_group"]
        if rho >= 0.60 and rmse <= 1.40:
            return "strong screening support"
        if rho >= 0.45 and rmse <= 1.60:
            return "contextual screening support"
        return "weak support; interpret cautiously"

    def use(row):
        label = str(row["task_label"])
        if label in {"LC50", "EC50"}:
            return "primary acute-priority evidence"
        if label in {"NOEC", "LOEC", "NOEL"}:
            return "supporting or endpoint-specific evidence"
        return "not used as a strong main-text driver"

    out = df[
        [
            "task_label",
            "n_candidates_scaffold_group",
            "rmse_log10_scaffold_group",
            "spearman_rho_scaffold_group",
            "r2_scaffold_group",
        ]
    ].copy()
    out["endpoint_reliability_tier"] = out.apply(tier, axis=1)
    out["recommended_reporting_use"] = out.apply(use, axis=1)
    return out


def formula_terms():
    rows = [
        {
            "score": "stage4_priority_score",
            "term": "toxicity_score_mean",
            "definition": "-mean predicted log10(mg L-1) ECOTOX-label signal",
            "coefficient": "1.00",
            "interpretation": "Higher values indicate a stronger predicted toxicity-label signal.",
        },
        {
            "score": "stage4_priority_score",
            "term": "occurrence_score",
            "definition": "log10(wqp_detected_records + 1) + 0.2 x log10(wqp_records + 1)",
            "coefficient": "1.00",
            "interpretation": "Public monitoring evidence; retained source column name is exposure_score, but the term is not a direct exposure concentration.",
        },
        {
            "score": "stage4_priority_score",
            "term": "has_fluorine",
            "definition": "1 for fluorine-containing candidate structures; 0 otherwise",
            "coefficient": "0.25",
            "interpretation": "Small scope-alignment term; sensitivity analysis removes this term.",
        },
        {
            "score": "stage4_priority_score",
            "term": "pred_log10_sd",
            "definition": "Bootstrap prediction standard deviation on the log10(mg L-1) scale",
            "coefficient": "-0.50",
            "interpretation": "Uncertainty penalty; wider bootstrap spread lowers the priority score.",
        },
        {
            "score": "precursor_integrated_score",
            "term": "precursor_motif_score",
            "definition": "Weighted structural motif score from CF3, perfluoroalkyl component, fluorine count, sulfonamide/sulfonate, labile linkage, and aromatic C-F features",
            "coefficient": "0.55",
            "interpretation": "Hypothesis-generating transformation-screening support only.",
        },
    ]
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=str(Path(__file__).resolve().parent.parent / "project_results_data"),
        help="Directory containing project_results_data outputs.",
    )
    args = parser.parse_args()

    results = Path(args.results_dir)
    out_dir = results / "priority_sensitivity"
    ensure_dir(out_dir)

    priority = pd.read_csv(results / "stage6_precursor_motifs" / "stage6_precursor_priority.tsv", sep="\t")
    scored = build_scores(priority)
    scored.to_csv(out_dir / "priority_sensitivity_scores_all_exposed.tsv", sep="\t", index=False)

    fluorinated = scored[scored["has_fluorine"].fillna(0).astype(int) == 1].copy()
    fluorinated["fluorinated_stage4_rank"] = rank_column(fluorinated, "score_stage4_recomputed")
    fluorinated["fluorinated_precursor_rank"] = rank_column(fluorinated, "score_precursor_integrated")
    fluorinated = fluorinated.sort_values("score_stage4_recomputed", ascending=False)
    fluorinated.to_csv(out_dir / "main_fluorinated_priority_queue.tsv", sep="\t", index=False)
    fluorinated.head(50).to_csv(out_dir / "main_fluorinated_priority_top50.tsv", sep="\t", index=False)

    non_f = scored[scored["has_fluorine"].fillna(0).astype(int) == 0].copy()
    non_f["nonfluorinated_context_rank"] = rank_column(non_f, "score_stage4_recomputed")
    non_f = non_f.sort_values("score_stage4_recomputed", ascending=False)
    non_f.to_csv(out_dir / "nonfluorinated_context_priority_rows.tsv", sep="\t", index=False)

    variants = [
        ("base_stage4", "score_stage4_recomputed", None),
        ("no_fluorination_term", "score_no_fluorination_term", None),
        ("detected_fraction_only", "score_fraction_only_occurrence", None),
        ("half_occurrence_weight", "score_half_occurrence_weight", None),
        ("hazard_uncertainty_only", "score_hazard_uncertainty_only", None),
        ("high_confidence_only", "score_stage4_recomputed", lambda d: d["priority_confidence"].astype(str).eq("high")),
        ("precursor_integrated", "score_precursor_integrated", None),
    ]
    variant_summary(fluorinated, variants).to_csv(out_dir / "priority_sensitivity_variant_summary.tsv", sep="\t", index=False)
    topk_overlap(fluorinated, variants).to_csv(out_dir / "priority_sensitivity_topk_overlap.tsv", sep="\t", index=False)

    metrics = pd.read_csv(results / "figure_source_data" / "table_endpoint_rdkit_random_vs_scaffold.tsv", sep="\t")
    endpoint_reliability(metrics).to_csv(out_dir / "endpoint_reliability_tiers.tsv", sep="\t", index=False)
    formula_terms().to_csv(out_dir / "priority_score_formula_terms.tsv", sep="\t", index=False)

    summary_rows = [
        {"metric": "all_exposed_priority_rows", "value": len(scored)},
        {"metric": "fluorinated_main_priority_rows", "value": len(fluorinated)},
        {"metric": "nonfluorinated_context_rows_excluded_from_main_queue", "value": len(non_f)},
        {"metric": "fluorinated_high_confidence_rows", "value": int((fluorinated["priority_confidence"] == "high").sum())},
        {"metric": "fluorinated_medium_confidence_rows", "value": int((fluorinated["priority_confidence"] == "medium").sum())},
        {"metric": "fluorinated_low_confidence_rows", "value": int((fluorinated["priority_confidence"] == "low").sum())},
    ]
    pd.DataFrame(summary_rows).to_csv(out_dir / "priority_queue_summary.tsv", sep="\t", index=False)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
