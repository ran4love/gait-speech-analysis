# Public Gait-Speech Analysis Code

This folder packages the manuscript workflow into a simple, reproducible structure for non-computer-science users.

The manuscript workflow uses MMPose for gait keypoint recognition, WhisperX for speech recognition, feature extraction for gait and Serial-7 speech, and joint gait-speech modeling for cognitive impairment screening.

## Quick Start

```bash
cd paper_public_code
cp config.example.ini config.ini
python run_pipeline.py --config config.ini --step validate
python run_pipeline.py --config config.ini --dry-run
```

Edit `config.ini`, then run one step:

```bash
python run_pipeline.py --config config.ini --step gait-features
python run_pipeline.py --config config.ini --step speech-features
python run_pipeline.py --config config.ini --step experiments
python run_pipeline.py --config config.ini --step paper-experiments
```

Run everything:

```bash
python run_pipeline.py --config config.ini --step all
```

## Workflow

```text
gait videos
  -> MMPose keypoint JSON
  -> gait features

speech wav files
  -> WhisperX transcript JSON
  -> Serial-7 speech features

gait features + speech task8/task9 features + demographic/label xlsx
  -> joint dataset
  -> classification and regression results

paper modeling table
  -> 5x10 repeated stratified CV model benchmark
  -> repeated stratified train-test split benchmark
  -> bootstrap 95% confidence intervals
  -> AdaBoost/RF/SVM(RBF)/DT x all-features/Nested-selected publication table
  -> nested training-fold-only feature-selection benchmark
  -> strict nested held-out model performance promoted as main classification output
  -> confusion matrices, ROC curves, and calibration curves
  -> age/education/age+sex+education confounder-adjusted AUROC stability analysis
  -> out-of-fold ROC curves
  -> SHAP importance and beeswarm figures
  -> repeated-K-fold regression table for MoCA/MMSE
```

## What Is Included

- User-facing wrapper: `run_pipeline.py`
- Editable config: `config.example.ini`
- Blank input templates: `data/templates/`
- Gait feature extraction: `src/gaitfeature.py`
- WhisperX recognition: `src/whisperx_recognize.py`
- Speech feature extraction: `src/speech_feature_extract.py`
- Exploratory experiment reproduction: `src/legacy_exploratory/analyze_dual_task_cost_gait_speech_from_json.py`
- Paper-level repeated-CV/ROC/SHAP reproduction: `src/paper_experiment_reproduction.py`
- Chinese guide: `docs/non_programmer_guide_zh.md`
- Minimal paper reproduction guide: `docs/minimal_paper_reproduction_zh.md`
- Environment notes: `docs/environment_notes.md`
- Code manifest: `docs/code_manifest.md`
- Feature-selection leakage audit: `docs/feature_selection_protocol.md`

## What Is Not Included

Videos, audio files, demographic tables, labels, and derived participant-level feature tables are not included. Put local data under `data/`.

The exact historical `batch_mmpose.sh` was not present in the local workspace. MMPose recognition is therefore exposed through `gait.mmpose_command_template` in `config.ini`; this lets users reuse a server batch script or an official MMPose command without editing Python code.

## Main Outputs

- `results/gait_features/gait_without_ds.json`
- `results/gait_features/gait_ds.json`
- `results/gait_features/gait_features_check_previous_format.xlsx`
- `results/speech_features/task8_features.json`
- `results/speech_features/task9_features.json`
- `results/experiments/joint_dataset.csv`
- `results/experiments/classification_results.csv`
- `results/experiments/regression_results.csv`
- `results/experiments/summary.json`
- `results/validation_report.json`
- `results/step_status.json`
- `results/paper_experiments/classification_repeated_cv.csv`
- `results/paper_experiments/classification_train_test_split_summary.csv`
- `results/paper_experiments/classification/model_performance_table.csv`
- `results/paper_experiments/classification/main_output_source.json`
- `results/paper_experiments/classification/confusion_matrices.png`
- `results/paper_experiments/classification/roc_curves.png`
- `results/paper_experiments/classification/calibration_curves.png`
- `results/paper_experiments/classification/nested_feature_selection/nested_feature_selection_performance.xlsx`
- `results/paper_experiments/classification/nested_feature_selection/nested_selected_features_by_fold.csv`
- `results/paper_experiments/classification/nested_feature_selection/nested_feature_selection_protocol.json`
- `results/paper_experiments/classification/confounder_adjustment/confounder_adjusted_performance_table.csv`
- `results/paper_experiments/classification/confounder_adjustment/auroc_stability_after_adjustment.csv`
- `results/paper_experiments/classification/confounder_adjustment/auroc_stability_after_adjustment.png`
- `results/paper_experiments/fig2_metric_bars_4panels.png`
- `results/paper_experiments/fig3_oof_roc_curves.png`
- `results/paper_experiments/fig4_shap_beeswarm_GBDT.png`
- `results/paper_experiments/regression/regression_repeated_cv.csv`
- `results/paper_experiments/regression/table2_regression_metric_bars.png`

Repeated summaries include `mean_*`, `std_*`, `ci95_low_*`, and `ci95_high_*` columns.

By default, the main classification table, ROC, calibration, and confusion matrix
are copied from nested feature-selection held-out predictions
(`promote_nested_selection = yes`). Fixed-panel results are retained as
`fixed_panel_*` files for audit.


## Recommended Public-Release Check

Before sharing publicly:

1. Confirm no participant videos/audio/xlsx/json are included.
2. Replace institution-specific server paths in examples with generic paths.
3. Add the final license selected by the manuscript team.
4. Run `python run_pipeline.py --config config.ini --dry-run` on a clean machine.
5. Review `docs/feature_selection_protocol.md` before reporting feature-selected model performance.
