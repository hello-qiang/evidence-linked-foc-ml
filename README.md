# Evidence-Linked FOC ML Analysis Code

This repository supports the study "Evidence-Linked Machine Learning Prioritizes Fluorinated Organic Contaminants for Monitoring and Transformation Screening".

It contains the core data-processing, machine-learning, validation, uncertainty, transferability, precursor-motif, and priority-screening scripts.

## Companion Data

The matched public data tables, model-ready inputs, completed model outputs, validation summaries, priority tables, and figure/table source data are available from Zenodo:

Tao, Q. et al. Evidence-Linked Machine Learning Data Package for Prioritizing Fluorinated Organic Contaminants. Zenodo. https://doi.org/10.5281/zenodo.20741547

## Scripts

| Script | Purpose |
|---|---|
| `prepare_stage1_tables.py` | Match and summarize WQP and ECOTOX records against the candidate universe. |
| `build_stage1_ml_tables.py` | Build candidate-level evidence features and ECOTOX endpoint-label tables. |
| `run_stage2_modeling.py` | Train baseline non-RDKit toxicity models. |
| `postprocess_stage2_predictions.py` | Apply applicability-domain screening and build baseline priority tables. |
| `run_stage3_rdkit_modeling.py` | Generate RDKit descriptors/fingerprints, scaffold groups, metrics, and predictions. |
| `run_stage4_robustness_uncertainty.py` | Run repeated scaffold validation and bootstrap uncertainty analysis. |
| `run_stage5_family_transfer.py` | Run chemical-family holdout validation and transfer-aware prioritization. |
| `run_stage6_precursor_motifs.py` | Screen fluorinated candidates for precursor-relevant motifs. |
| `run_priority_sensitivity.py` | Build the fluorinated-only main priority queue, nonfluorinated context table, score formula table, endpoint reliability tiers, and ranking-sensitivity summaries from stage 6 outputs. |
| `requirements.txt` | Python package requirements. |
| `file_manifest.tsv` | File size, line count, and SHA-256 checksums for tracked files in this repository, excluding the manifest itself. |

## Runtime Layout

Use a `--base` path with this layout:

```text
project_base/
  03_chemical_universe/
    candidate_inventory.tsv
    candidate_inventory_pubchem_enriched.tsv
  04_endpoint_labels/
    ecotox_candidate_records.tsv
    ecotox_candidate_summary.tsv
  05_occurrence_exposure/
    wqp/
      wqp_candidate_occurrence.tsv
      wqp_candidate_occurrence_summary.tsv
  07_outputs/
    stage1_data_package/
    stage2_modeling/
    stage3_rdkit_modeling/
    stage4_robustness_uncertainty/
    stage5_family_transfer/
    stage6_precursor_motifs/
```

For rerunning stages 2-6 from the companion data package, create the runtime layout and copy the curated model-input tables:

```bash
mkdir -p /path/to/project_base/07_outputs/stage1_data_package
cp /path/to/raw_public_data/curated_model_inputs/*.tsv /path/to/project_base/07_outputs/stage1_data_package/
```

To rebuild the stage-1 model-input package from the matched public source tables, place the source tables in the corresponding runtime folders:

```bash
mkdir -p /path/to/project_base/03_chemical_universe
mkdir -p /path/to/project_base/04_endpoint_labels
mkdir -p /path/to/project_base/05_occurrence_exposure/wqp
cp /path/to/raw_public_data/source_public_tables/candidate_inventory*.tsv /path/to/project_base/03_chemical_universe/
cp /path/to/raw_public_data/source_public_tables/ecotox_candidate_*.tsv /path/to/project_base/04_endpoint_labels/
cp /path/to/raw_public_data/source_public_tables/wqp_candidate_occurrence*.tsv /path/to/project_base/05_occurrence_exposure/wqp/
```

The companion `project_results_data/` folder contains completed outputs for stages 2-6 and can be used directly to inspect or regenerate derived priority-sensitivity tables.

## Environment

A conda-forge environment is recommended for RDKit:

```bash
conda env create -f environment.yml
conda activate evidence-linked-foc-ml
```

## Recommended Run Order

Most users should start from the curated model-input tables in `raw_public_data/curated_model_inputs/`:

```bash
python run_stage2_modeling.py --base /path/to/project_base --n-jobs 4
python postprocess_stage2_predictions.py --base /path/to/project_base
python run_stage3_rdkit_modeling.py --base /path/to/project_base --n-jobs 4
python run_stage4_robustness_uncertainty.py --base /path/to/project_base --n-jobs 4
python run_stage5_family_transfer.py --base /path/to/project_base --n-jobs 4
python run_stage6_precursor_motifs.py --base /path/to/project_base
python run_priority_sensitivity.py --results-dir /path/to/project_results_data
```

To rebuild the curated model-input tables from the matched public source tables before modeling, first run:

```bash
python build_stage1_ml_tables.py --base /path/to/project_base
```

The lower-level `prepare_stage1_tables.py` script is retained for provenance and can be used when the original WQP result ZIPs, selected WQP term table, and ECOTOX archive inputs are available in the runtime layout. The companion data package already includes the matched public source tables and curated model-input tables, so this lower-level matching step is not required for reproducing the modeling outputs.

Python dependencies are also listed in `requirements.txt` for environments that do not use conda.

## License

This code is released under the MIT License.
