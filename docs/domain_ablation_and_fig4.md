# Domain Ablation and Fig. 4 Reproduction

This document describes the public code used to reproduce the full-cohort
within-task domain ablation and Fig. 4.

## Purpose

The analysis evaluates how different feature domains contribute to
classification of cognitive impairment status in the same dual-task TUG-DT
cohort. It is not external validation and not task-only validation.

The five feature sets are:

| Feature set | Included domains |
| --- | --- |
| `demographics-only` | age, sex/gender, education |
| `gait-only` | dual-task gait features |
| `speech-only` | dual-task Serial-7 speech features |
| `gait + speech` | gait and speech domains |
| `gait + speech + demographics` | gait, speech and demographics |

## Leakage-Controlled Protocol

For each feature set and model, the code uses repeated stratified five-fold
cross-validation with 10 repeats. In every outer fold:

1. The training fold alone is used for feature ranking.
2. Ranking is based on `abs(training_univariate_AUROC - 0.5)`.
3. The selected feature count is `min(select_k_cap, available_features)`.
4. Median imputation is fitted inside the model pipeline using the training
   fold only.
5. The held-out fold is used only for prediction and metric calculation.

The held-out fold is not used for feature ranking, top-k selection, imputation
fitting, scaling fitting or model fitting.

## Required Input Table

Use `data/templates/paper_modeling_dataset_template.xlsx` as the schema. The
table must include:

- `ID`
- binary label column, default `label`
- demographics: `age`, `edu`, and either `sex` or `gender`
- gait features listed in `src/domain_ablation_analysis.py`
- speech features listed in `src/domain_ablation_analysis.py`

Protected videos, audio, transcripts and participant-level data are not included
in the public repository.

## Commands

Run the main paper experiments first if Fig. 4 will include RF SHAP contribution
and nested feature recurrence:

```bash
python run_pipeline.py --config config.ini --step paper-experiments
```

Run domain ablation:

```bash
python run_pipeline.py --config config.ini --step domain-ablation
```

Generate Fig. 4:

```bash
python run_pipeline.py --config config.ini --step figures
```

Or run the underlying scripts directly:

```bash
python scripts/run_domain_ablation.py --config config.ini
python scripts/make_paper_figures.py --config config.ini
```

## Domain Ablation Outputs

Default output directory: `results/domain_ablation/`

| File | Purpose |
| --- | --- |
| `domain_ablation_performance_table.csv/.xlsx` | Mean, SD and bootstrap 95% CI for AUROC, ACC, precision, recall and F1. |
| `domain_ablation_fold_metrics.csv` | Per-fold held-out metrics. |
| `domain_ablation_predictions.csv` | Held-out predictions. |
| `domain_ablation_confusion_matrix_summary.csv` | Aggregated TN, FP, FN and TP counts. |
| `domain_ablation_feature_manifest.csv` | Requested-to-actual feature-column mapping. |
| `domain_ablation_selected_feature_frequency.csv` | Training-fold nested feature-selection recurrence. |
| `domain_ablation_paired_delta_auroc.csv/.xlsx` | Same-fold paired AUROC differences. |
| `domain_ablation_protocol.json` | Machine-readable protocol and QC summary. |
| `domain_ablation_results_workbook.xlsx` | Combined workbook for review. |
| `domain_ablation_auroc_bar.svg/.png` | Domain ablation AUROC figure. |
| `domain_ablation_paired_delta_auroc.svg/.png` | Paired Delta AUROC figure. |
| `domain_ablation_metric_heatmap.svg/.png` | Metric heatmap. |
| `domain_ablation_roc_curves.svg/.png` | ROC curves. |

## Fig. 4 Outputs

Default output directory: `results/figures/`

Fig. 4 is titled:

```text
Fig. 4 | Model contribution and interpretability.
```

Panels:

```text
a Domain ablation AUROC
b Paired Delta AUROC
c RF feature contribution
d Nested feature recurrence
```

The Fig. 4 generator uses:

- `results/domain_ablation/domain_ablation_performance_table.csv`
- `results/domain_ablation/domain_ablation_fold_metrics.csv`
- `results/domain_ablation/domain_ablation_paired_delta_auroc.csv`
- `results/paper_experiments/shap_importance_Random_Forest.csv`
- `results/paper_experiments/classification/nested_feature_selection/nested_selected_feature_frequency.csv`

## Interpretation Boundary

If `demographics-only` performs close to `gait + speech`, report that
classification performance is strongly influenced by demographic structure,
particularly education. Avoid interpreting the result as evidence that gait or
speech features alone are independent of demographics.
