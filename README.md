# ADualNet Research Toolkit

[**Launch the interactive synthetic demo**](https://ran4love.github.io/gait-speech-analysis/) ·
[Full local workflow](#3-run) · [MIT License](LICENSE)

**For editors and readers:** the online demo needs no account or installation.
It uses mathematical pose coordinates and a scripted transcript, not patient
recordings. Review response errors, adjust speech-task annotations, confirm both
review sections and download XLSX/CSV/JSON. It makes no application API calls,
does not accept real-data uploads and does not run MMPose, Whisper or a classifier.
Gait references are transparently precomputed; reviewed speech features are
recomputed in your browser. See [public demo scope and privacy](docs/public_demo.md).

A local browser workbench for trying the TUG-DT gait-speech research paradigm.
Start with one synchronized video: MMPose pose estimation + Whisper transcription,
human review, then versioned feature/QC export. A separate feature-table workspace
supports cohort modelling. No participant data or pretrained classifier is distributed.

## 1. Prepare Your Inputs

Try **体验人工核查演示** first; no input files or recognition models are needed.
For real recognition, provide one video with synchronized task audio (up to 512 MB,
10 minutes by default), a local pseudonymous ID and your task/scoring protocol.
An environment administrator configures `recognition.local.json`; use either
`recognition.example.json` for local inference or `recognition.ssh.example.json`
for your own Ubuntu SSH server. Never put passwords in these files.

See the [single-video walkthrough](docs/video_workflow_zh.md). For cohort modelling, use
`data/templates/toolkit_features_template.csv`: one row per participant, a unique
`ID`, binary `label`, 3 demographic, 15 gait and 8 speech-task features. Record
your measurement version, units, task/scoring rules and the meaning of label 0/1.
Do not use the legacy templates containing both `sex` and `gender` for this UI.

See the [Chinese walkthrough](docs/toolkit_guide_zh.md) and
[statistical protocol](docs/toolkit_statistical_protocol.md). Raw recording
recognition and historical manuscript reproduction have separate entrypoints
and provenance requirements described below.

Before combining exports or reusing classifier results, see the
[measurement profiles](docs/measurement_profiles.md). The reviewed-video export,
the generic26-predictor table interface, the historical25-predictor primary model
and the separate7+6+2 temporal audit are not interchangeable. A read-only check is:

    python adualnet.py contract-check --input reviewed_features.csv --profile video_review_v1

This does not fit a model or authorize cohort admission. It also does not turn
undefined contact-phase fields into imputable measurements.

## 2. Install The CPU Toolkit

Use Python 3.11 or 3.12 in a dedicated environment:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows instead: .venv\Scripts\activate
python -m pip install -r requirements-toolkit.txt
```

The feature-table demo does not require a GPU, MMPose, WhisperX or SHAP.
Install FFmpeg/ffprobe for video handling. Recognition dependencies belong in
separate compatible environments; the Mac/client need not install MMPose or
Whisper when using Ubuntu SSH inference. Local video recognition additionally
uses `requirements-video.txt` and a separately installed MMPose/MMCV/PyTorch stack.
See [video setup](docs/video_workflow_zh.md) and [environment notes](docs/environment_notes.md).

## 3. Run

```bash
python adualnet.py serve
```

Open `http://127.0.0.1:8765`. Under **从一个视频开始**, select **体验人工核查演示**
or upload your video after checking the recognition environment. Review the
participant track, task windows, keypoints and each spoken response, record the
scoring rule, then explicitly confirm both gait and speech before exporting XLSX/CSV.
Drafts, raw recognition and previous reviews/exports are retained separately.

The header's **English / 中文** buttons switch the interface without reloading or
discarding edits. English is the initial default; your choice is remembered locally.
User-authored transcripts, scoring rules, notes, IDs, raw logs and exported data
are not translated. Existing aggregate reports retain their original language.

Version 0.3 uses a navy-and-white research workspace: persistent navigation,
compact task context, a job-driven four-step indicator, separate recording and
environment cards, and a **Go to review** shortcut. Technical logs are collapsed
by default. This interface redesign does not change recognition settings,
measurement formulas or classification methods. See the
[workspace design notes](docs/medical_workspace_design.md).

The **模型演示** tab is a separate synthetic 3-fold × 2-repeat modelling run;
own-table runs use 5 folds × 10 repeats. Video exports are not automatically sent
to a classifier or certified as the historical manuscript measurement panel.

CLI equivalents:

```bash
python adualnet.py video-demo --out local_workspace/video_demo_v1
python adualnet.py video --input my_video.mp4 --out local_workspace/video_v1 \
  --recognition-config recognition.local.json
python adualnet.py demo
python adualnet.py validate --input my_features.xlsx --sheet info
python adualnet.py analyze --input my_features.xlsx --sheet info \
  --out local_workspace/runs/my_study_v1 \
  --measurement-version siteA_reviewed_v1 \
  --outcome-definition "0 = unimpaired; 1 = impaired, according to study protocol"
```

New reports contain domain ablation, demographic components, paired AUROC
differences, Brier/ECE, descriptive calibration and education strata. Feature
ranking, imputation and model fitting use training folds only. All domains use
identical participants and splits. Repeated predictions stay together in
participant bootstrap; intervals are conditional on cached models and do not
fully capture training/selection uncertainty. The new estimand is mean
complete-repeat OOF performance, which differs from historical mean-fold scores.

`local_workspace/` holds sensitive video/audio, transcripts, manual reviews and
participant-level outputs; keep it local. In SSH mode normalized video, audio and
recognition settings are also sent to, and retained on, your configured server.
Model downloads may contact upstream registries; study media are not sent to an
external ASR API. Browser upload requires processing/transfer acknowledgement.
Do not expose the loopback web server publicly. Cohort report HTML/XLSX are aggregate but can contain study
metadata you supplied. Review those before sharing. Build a source-only GitHub
release candidate with `python scripts/build_public_toolkit.py`. It excludes
study data, local runs, checkpoints and Git history. The source is MIT-licensed;
separate dependencies/weights retain their own terms. See [license notice](LICENSE_NOTICE.md).

## Video Measurement Boundaries

The video workflow reuses the existing ankle-event gait extractor, with retained
frame slots, confidence/gap QC and no diagnosis-dependent phase presets. Spatial
outputs are image-pixel estimates, not calibrated metric step lengths. Step
length/time are half their stride estimates by construction, not independent
physiological validation; step/stride asymmetry are consequently not independent.
The heuristic events are not validated heel strikes. `stance_times`, `swing_times`
and `double_support_times` remain explicitly missing: no validated contact-phase
definition is implemented here. Missing values and failed QC are not repaired by
fabricated constants. Do not claim clinical accuracy from a software smoke test.

Speech features use human-reviewed response onsets and explicit correctness labels;
the model is not prompted with an expected subtraction sequence. Full-text edits
are audit notes; counts come from the reviewed response rows. No clinician-certified
diagnostic or screening model is bundled with this video workflow.

## Historical Scripts And Provenance

## Manuscript-specific provenance notice

The 2026-10-30 audit separates historical table-based model reproduction from
later raw-recording extraction utilities. The latter are not certified to
regenerate the original 164-person manuscript feature table. Historical gait
time/cadence names, phase versions and speech annotation conventions require
the qualifications in [manuscript feature provenance](docs/manuscript_feature_provenance.md).
The historical AUROC must not be attributed to corrected inputs. A later,
separately registered measurement/QC reevaluation used a different temporal panel
and cohort; it is not a performance validation of this workbench release.
The read-only check in that document does not train or score models and leaves
all historical sources intact.

This folder contains saved-table modeling scripts and later gait/speech extraction
utilities in a common structure. Reproduction from the historical feature table
does not establish end-to-end reproduction from raw recordings.

The extraction utilities use MMPose keypoints and WhisperX transcripts. Their
measurement definitions must be reconciled with the study-specific feature
contract before new outputs are used in manuscript analyses.

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
python run_pipeline.py --config config.ini --step domain-ablation
python run_pipeline.py --config config.ini --step figures
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
  -> full-cohort within-task domain ablation
  -> paired fold-level Delta AUROC
  -> Fig. 4 model contribution and interpretability
  -> Python-only baseline, group-difference and correlation statistics
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
- Full-cohort within-task domain ablation: `src/domain_ablation_analysis.py`
- Fig. 4 model contribution and interpretability: `src/figure_model_contribution_interpretability.py`
- Python-only statistical analysis: `src/statistical_analysis_python.py`
- Chinese guide: `docs/non_programmer_guide_zh.md`
- Minimal paper reproduction guide: `docs/minimal_paper_reproduction_zh.md`
- Environment notes: `docs/environment_notes.md`
- Code manifest: `docs/code_manifest.md`
- Feature-selection leakage audit: `docs/feature_selection_protocol.md`
- Domain ablation and Fig. 4 guide: `docs/domain_ablation_and_fig4.md`

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
- `results/paper_experiments/classification/table2_classification_performance.xlsx`
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
- `results/domain_ablation/domain_ablation_performance_table.csv`
- `results/domain_ablation/domain_ablation_paired_delta_auroc.csv`
- `results/domain_ablation/domain_ablation_protocol.json`
- `results/domain_ablation/domain_ablation_auroc_bar.svg`
- `results/domain_ablation/domain_ablation_paired_delta_auroc.svg`
- `results/figures/fig4_model_contribution_interpretability.svg`
- `results/figures/fig4_model_contribution_interpretability.pdf`
- `results/figures/fig4_model_contribution_interpretability.png`
- `results/figures/fig4_model_contribution_interpretability.tiff`
- `results/statistical_analysis/statistical_analysis_results.xlsx`
- `results/statistical_analysis/figS_statistical_correlation_group_combined.svg`
- `results/statistical_analysis/figS_statistical_group_analysis.svg`
- `results/statistical_analysis/figS_correlation_analysis.svg`

Repeated summaries include `mean_*`, `std_*`, `ci95_low_*`, and `ci95_high_*` columns.

By default, the main classification table, ROC, calibration, and confusion matrix
are copied from nested feature-selection held-out predictions
(`promote_nested_selection = yes`). Fixed-panel results are retained as
`fixed_panel_*` files for audit.


## Recommended Public-Release Check

Before sharing publicly:

1. Confirm no participant videos/audio/xlsx/json are included.
2. Replace institution-specific server paths in examples with generic paths.
3. Retain the MIT license and applicable third-party notices.
4. Run `python run_pipeline.py --config config.ini --dry-run` on a clean machine.
5. Review `docs/feature_selection_protocol.md` before reporting feature-selected model performance.
