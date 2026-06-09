# Code Manifest

## User-Facing Entry Points

| File | Purpose |
| --- | --- |
| `run_pipeline.py` | One-command workflow runner. |
| `config.example.ini` | Editable settings for paths, MMPose, WhisperX, feature extraction, and experiment reproduction. |
| `scripts/validate_inputs.py` | Checks config paths, dependencies, table columns, and writable output directories. |
| `scripts/run_gait_recognition_mmpose.py` | Runs MMPose through a command template. |
| `scripts/extract_gait_features.py` | Runs `src/gaitfeature.py` on MMPose keypoint JSON. |
| `scripts/merge_gait_json_to_xlsx.py` | Converts gait JSON outputs to `check_previous`-format xlsx for experiment reproduction. |
| `scripts/run_speech_recognition_whisperx.py` | Runs WhisperX ASR for TASK08 and TASK09 audio. |
| `scripts/extract_speech_features.py` | Extracts Serial-7 speech metrics from WhisperX JSON. |
| `scripts/reproduce_experiments.py` | Reproduces the exploratory gait-speech joint analysis through `src/legacy_exploratory/`. |
| `scripts/reproduce_paper_experiments.py` | Reproduces manuscript-level model benchmark, nested feature-selection benchmark, OOF ROC, and SHAP outputs. |
| `scripts/qc_gait_xlsx.py` | Builds a physics-informed QC xlsx report. |

## Core Source Code

| File | Purpose |
| --- | --- |
| `src/gaitfeature.py` | Merged gait feature extractor with corrected physical rules. |
| `src/whisperx_recognize.py` | WhisperX ASR script with task-specific presets. |
| `src/speech_feature_extract.py` | Serial-7 speech feature extractor. |
| `src/paper_experiment_reproduction.py` | Paper-level repeated-CV classification, AdaBoost/RF/SVM(RBF)/DT all-features/Nested-selected publication table, nested training-fold-only feature selection, confusion matrix, ROC, calibration curve, confounder adjustment, bootstrap CI, SHAP, and separate regression reproduction. |
| `src/analyze_new_data_check.py` | Lightweight ML and cross-validation utilities. |
| `src/qc_check_previous_2026.py` | Gait xlsx QC rules and report builder. |
| `src/compare_gait_metrics_to_xlsx.py` | Minimal xlsx read/write helpers and metric comparison utilities. |
| `src/legacy_exploratory/analyze_dual_task_cost_gait_speech_from_json.py` | Exploratory gait-speech analysis using task8/task9 JSON features; not the leakage-controlled main paper benchmark. |
| `src/legacy_exploratory/analyze_matched_pred2025_gait_vs_demo.py` | Exploratory gait-demographic analysis helpers. |
| `src/legacy_exploratory/analyze_matched_pred2025_by_task.py` | Exploratory single/dual gait task analysis helpers. |

## Data Templates

| File | Purpose |
| --- | --- |
| `data/templates/paper_modeling_dataset_template.xlsx/.csv` | Fixed-column table for minimal paper reproduction. |
| `data/templates/demo_template.xlsx/.csv` | Demographic/label template for the full gait-speech workflow. |
| `data/templates/gait_features_check_previous_template.xlsx/.csv` | `check_previous`-format gait feature table template. |

## Data Not Included

The public package does not include videos, audio, patient demographics, labels, or feature tables derived from protected study data. Users should place their own files under `data/`.

## Protocol Documents

| File | Purpose |
| --- | --- |
| `docs/feature_selection_protocol.md` | Documents Nested-selected feature provenance, historical ADual-selected fallback, nested feature-selection outputs, and leakage-control rules. |
| `docs/minimal_paper_reproduction_zh.md` | Chinese guide for reproducing paper models from one `paper_modeling_dataset.xlsx`. |
