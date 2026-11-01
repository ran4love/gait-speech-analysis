# Gait-Speech Workflow Inventory

Generated: 2026-06-07

This document inventories the current gait, speech, and gait-speech analysis workflows in this workspace. It focuses on code that exists now, the inputs it expects, the outputs already present, and the practical order for rerunning the workflow.

## 1. Current Workflow Map

| Stage | Purpose | Main code | Current outputs |
|---|---|---|---|
| Demographic baseline | Analyze `new data_check.xlsx` labels, MoCA/MMSE, demographics, baseline ML | `analyze_new_data_check.py` | `new_data_check_analysis/` |
| Gait feature extraction | Convert MMPose keypoint JSON/video into gait features | `gaitfeature.py` | `gait_features_*.json`, `check_previous_format_*.xlsx` |
| Gait QC | Score gait metrics by physical consistency rules | `qc_check_previous_2026.py` | `*_qc.xlsx`, `gait_qc_rules_v1.md` |
| Single vs dual gait | Pair same ID single/dual gait records and compute deltas | `analyze_single_dual_gait_features.py`, `analyze_single_dual_from_xlsx.py` | `single_dual_analysis/`, `gait_tele_single_dual_analysis/` |
| Matched gait vs demo | Match gait rows to demographics and run group/ML analyses | `legacy_exploratory/analyze_matched_pred2025_by_task.py`, `analyze_matched_pred_pooled_by_task.py`, `legacy_exploratory/analyze_matched_pred2025_gait_vs_demo.py` | `matched_pred*_analysis*/`, `matched_pred_pooled_by_task_analysis_fast/` |
| Speech + gait cost | Compute gait dual-task cost and speech task8/task9 cost, then model label/MoCA | `legacy_exploratory/analyze_dual_task_cost_gait_speech_from_json.py`, `analyze_dual_task_cost_gait_speech.py` | `dual_task_cost_gait_speech_json_analysis/`, `dual_task_cost_gait_speech_analysis*/` |
| Summary report | Build integrated HTML report from gait and gait-speech results | `build_gait_summary_report.py` | `gait_summary_report_20260312/gait_speech_summary_report.html` |
| Calibration/QC experiments | Test camera calibration impact on tele/pred2026 gait QC | `gaitfeature.py`, ad hoc comparison scripts | `tele_calibra_qc_comparison.xlsx`, `pred2026_calibshot_qc_comparison.xlsx`, strategy docs |
| Recovery/rerun experiments | Recover missing pred2026 JSON, rerun suspicious gait rows, compare min stride settings | `rerun_*.py`, `simulate_event_detection_sweeps.py`, `analyze_event_detection_failures.py` | `rerun_*`, `event_detection_*`, `merged_gait_pred_2025_2026_*` |

## 2. Canonical End-to-End Gait-Speech Flow

### Step A. Build or refresh gait feature tables

Code:

- `gaitfeature.py`
- Legacy sources preserved as references: `gait_features_extraction_without_ds_new.py`, `gait_features_extraction_ds.py`

Typical commands:

```bash
python gaitfeature.py without_ds_json \
  --json-directory <mmpose_json_flat_dir> \
  --output-json-path gait_features_<dataset>_without_ds_adaptive.json \
  --frame-rate 30 \
  --distance-meters 6

python gaitfeature.py ds \
  --keypoints-directory <mmpose_json_flat_dir> \
  --new-features-file gait_features_<dataset>_ds_adaptive.json \
  --frame-rate 30 \
  --group HC
```

Important options:

- `--min-stride-time-sec`: currently present in `gaitfeature.py`, used to reduce suspicious short stride events.
- `--camera-calib-npz`: optional calibration/undistortion path. Existing experiments show calibration did not improve full-set QC for `gait_tele` or `pred_2026` with current calibration files.

Current key gait feature artifacts:

- `gait_features_pred_2025_without_ds_adaptive.json`
- `gait_features_pred_2025_ds_adaptive.json`
- `gait_features_pred_2026_without_ds_adaptive.json`
- `gait_features_pred_2026_ds_adaptive.json`
- `check_previous_format_pred_2025_adaptive.xlsx`
- `check_previous_format_new_2026_adaptive.xlsx`

### Step B. QC gait tables

Code:

- `qc_check_previous_2026.py`

Typical command:

```bash
python qc_check_previous_2026.py \
  --input-xlsx check_previous_format_new_2026_adaptive.xlsx \
  --output-xlsx check_previous_format_new_2026_adaptive_qc.xlsx
```

QC rules include:

- stride/step length ratio
- stride/step time ratio
- cadence formula consistency
- stance ratio
- double support ratio
- double support <= stance
- step/stride asymmetry
- walking speed range

Current major QC artifacts:

- `check_previous_format_pred_2025_adaptive_qc.xlsx`
- `check_previous_format_new_2026_adaptive_qc.xlsx`
- `check_previous_format_pred2026_baseline_qc.xlsx`
- `check_previous_format_pred2026_shot8_scaled_qc.xlsx`
- `check_previous_format_gait_tele_adaptive_qc.xlsx`

### Step C. Analyze single-task vs dual-task gait

Code:

- `analyze_single_dual_gait_features.py`: consumes gait feature JSON files.
- `analyze_single_dual_from_xlsx.py`: consumes a `check_previous_format_*.xlsx` table.

Default command for JSON-based pred2025/pred2026:

```bash
python analyze_single_dual_gait_features.py \
  --pred2025-without-ds gait_features_pred_2025_without_ds_adaptive.json \
  --pred2025-ds gait_features_pred_2025_ds_adaptive.json \
  --pred2026-without-ds gait_features_pred_2026_without_ds_adaptive.json \
  --pred2026-ds gait_features_pred_2026_ds_adaptive.json \
  --out-dir single_dual_analysis
```

Default command for tele xlsx:

```bash
python analyze_single_dual_from_xlsx.py \
  --xlsx check_previous_format_gait_tele_adaptive.xlsx \
  --out-dir gait_tele_single_dual_analysis
```

Current output directories:

- `single_dual_analysis/`
- `gait_tele_single_dual_analysis/`

Key current finding from `single_dual_analysis/summary_all.json`:

- `pred_2025` has 84 same-ID single/dual pairs.
- strongest deltas include `walking_speed`, `swing_times`, `stance_times`, `double_support_times`.
- `pred_2026` local early table had no single/dual paired IDs in that older run; later pooled runs use larger `new_2026` data.

### Step D. Match gait with demographics

Code:

- `analyze_new_data_check.py`
- `legacy_exploratory/analyze_matched_pred2025_gait_vs_demo.py`
- `legacy_exploratory/analyze_matched_pred2025_by_task.py`
- `analyze_matched_pred_pooled_by_task.py`

Current demographic input:

- `../new data_check.xlsx`

Typical pooled command:

```bash
python analyze_matched_pred_pooled_by_task.py \
  --demo-xlsx ../new\ data_check.xlsx \
  --gait-xlsx check_previous_format_pred_2025_adaptive.xlsx check_previous_format_new_2026_adaptive.xlsx \
  --out-dir matched_pred_pooled_by_task_analysis \
  --regression-target MoCA
```

Current important output directories:

- `new_data_check_analysis/`
- `matched_pred2025_gait_analysis/`
- `matched_pred2025_gait_analysis_moca/`
- `matched_pred2025_by_task_analysis/`
- `matched_pred_pooled_by_task_analysis/`
- `matched_pred_pooled_by_task_analysis_fast/`

Current pooled-fast summary:

- `matched_pred_pooled_by_task_analysis_fast/single`: 162 matched single-task rows.
- `matched_pred_pooled_by_task_analysis_fast/dual`: 309 matched dual-task rows.
- Single-task gait currently has stronger modeling performance than dual-task gait in the integrated report.

### Step E. Build speech dual-task cost and joint gait-speech models

There are two versions.

#### E1. Older speech-xlsx workflow

Code:

- `analyze_dual_task_cost_gait_speech.py`

Inputs:

- `../new data_check.xlsx`
- `check_previous_format_pred_2025_adaptive.xlsx`
- `check_previous_format_new_2026_adaptive.xlsx`
- speech xlsx such as `20260120HL-ASR-v2-89-0306.xlsx`
- sheet `TASK08-09`

Default command:

```bash
python analyze_dual_task_cost_gait_speech.py \
  --demo-xlsx ../new\ data_check.xlsx \
  --gait-xlsx check_previous_format_pred_2025_adaptive.xlsx check_previous_format_new_2026_adaptive.xlsx \
  --speech-xlsx data/speech_features/task08_09_features.xlsx \
  --speech-sheet TASK08-09 \
  --out-dir dual_task_cost_gait_speech_analysis \
  --target MoCA
```

Existing variants:

- `dual_task_cost_gait_speech_analysis/`
- `dual_task_cost_gait_speech_analysis_task0809/`
- `dual_task_cost_gait_speech_analysis_task0809_compare_old0306/`
- `dual_task_cost_gait_speech_analysis_task0809_compare_new0421_endsec/`

Status:

- These xlsx-based runs only matched 14-15 rows and all were `label=1`, so classification was not meaningful.
- They are historical comparison branches, not the strongest current gait-speech workflow.

#### E2. Current task8/task9 JSON workflow

Code:

- `legacy_exploratory/analyze_dual_task_cost_gait_speech_from_json.py`

Inputs:

- `../new data_check.xlsx`
- `check_previous_format_pred_2025_adaptive.xlsx`
- `check_previous_format_new_2026_adaptive.xlsx`
- `task8_features_rerun_20260311.json`
- `task9_features_rerun_20260311.json`

Default command:

```bash
python legacy_exploratory/analyze_dual_task_cost_gait_speech_from_json.py \
  --demo-xlsx ../new\ data_check.xlsx \
  --gait-xlsx check_previous_format_pred_2025_adaptive.xlsx check_previous_format_new_2026_adaptive.xlsx \
  --task8-json task8_features_rerun_20260311.json \
  --task9-json task9_features_rerun_20260311.json \
  --out-dir dual_task_cost_gait_speech_json_analysis \
  --target MoCA \
  --max-selected-features 3 \
  --top-candidates 6
```

Outputs:

- `dual_task_cost_gait_speech_json_analysis/joint_dataset.csv`
- `dual_task_cost_gait_speech_json_analysis/group_analysis.csv`
- `dual_task_cost_gait_speech_json_analysis/classification_results.csv`
- `dual_task_cost_gait_speech_json_analysis/regression_results.csv`
- `dual_task_cost_gait_speech_json_analysis/summary.json`

Current status:

- `n_joint_rows`: 115
- labels: 66 label0, 49 label1
- current strongest classification: speech-cost-only logistic regression, AUC about 0.736
- current strongest regression: speech-cost-only random forest, R2 about 0.059
- joint gait+speech has not beaten speech-only in the current run.

Speech feature set:

- `ACC`
- `Correct_persec`
- `Correct_Num`
- `Num`
- `Timecount`
- `Timecount_1stNum`
- `Num_20s`
- `CorrectNum_20s`

Gait cost features:

- All `GAIT_FEATURES` imported from `legacy_exploratory/analyze_matched_pred2025_gait_vs_demo.py`
- Cost definition in current scripts: `(dual - single) / single`

### Step F. Build integrated report

Code:

- `build_gait_summary_report.py`

Command:

```bash
python build_gait_summary_report.py
```

Output:

- `gait_summary_report_20260312/gait_speech_summary_report.html`

Inputs consumed by the report:

- `new_data_check_analysis/summary.json`
- `single_dual_analysis/pred_2025_summary.json`
- `matched_pred_pooled_by_task_analysis_fast/summary.json`
- `dual_task_cost_gait_speech_json_analysis/summary.json`
- `dual_task_cost_gait_speech_json_analysis/group_analysis.csv`
- `dual_task_cost_gait_speech_json_analysis/classification_results.csv`
- `dual_task_cost_gait_speech_json_analysis/regression_results.csv`
- `figures_sci/Figure*.svg`

Dependency note:

- `build_gait_summary_report.py` imports `pandas`; the current default local `python3` environment did not report `pandas` as installed during this inventory. Use the project environment where the report was previously generated, or install pandas before rerunning.

## 3. Code Inventory

| Code | Role | Inputs | Outputs / Notes |
|---|---|---|---|
| `gaitfeature.py` | Main merged gait feature extractor with corrected physical rules, min stride threshold, optional camera calibration | MMPose JSON/video folders | gait feature JSON files |
| `gait_features_extraction_without_ds_new.py` | Legacy non-dual-task extraction source | video + keypoint JSON | kept as reference |
| `gait_features_extraction_ds.py` | Legacy dual-task phase feature source | keypoint JSON | kept as reference |
| `qc_check_previous_2026.py` | Physics-informed QC report builder | gait xlsx | `*_qc.xlsx` |
| `compare_gait_metrics_to_xlsx.py` | xlsx parser/writer and pred2025/pred2026/previous comparison | gait JSON/xlsx | comparison xlsx |
| `compare_same_id_adaptive.py` | Same-ID pred2025 vs pred2026 adaptive comparison | adaptive gait xlsx | `same_id_comparison_pred2025_vs_new2026_adaptive.xlsx` |
| `analyze_single_dual_gait_features.py` | JSON-based single/dual gait delta analysis | gait feature JSON | `single_dual_analysis/` |
| `analyze_single_dual_from_xlsx.py` | xlsx-based single/dual gait delta analysis | gait xlsx | `gait_tele_single_dual_analysis/` |
| `analyze_new_data_check.py` | Demographic/label/MoCA baseline analysis | `../new data_check.xlsx` | `new_data_check_analysis/` |
| `legacy_exploratory/analyze_matched_pred2025_gait_vs_demo.py` | Match pred2025 gait to demographics | gait xlsx + demo xlsx | matched gait analysis |
| `legacy_exploratory/analyze_matched_pred2025_by_task.py` | Split matched pred2025 gait into single/dual task analyses | pred2025 gait xlsx + demo xlsx | `matched_pred2025_by_task_analysis/` |
| `analyze_matched_pred_pooled_by_task.py` | Pool pred2025 and pred2026 gait, then analyze by single/dual task | two gait xlsx + demo xlsx | `matched_pred_pooled_by_task_analysis*/` |
| `analyze_dual_task_cost_gait_speech.py` | Older xlsx speech workflow | speech xlsx sheet + gait xlsx + demo xlsx | `dual_task_cost_gait_speech_analysis*/` |
| `legacy_exploratory/analyze_dual_task_cost_gait_speech_from_json.py` | Current task8/task9 JSON speech workflow | task8/task9 JSON + gait xlsx + demo xlsx | `dual_task_cost_gait_speech_json_analysis/` |
| `build_gait_summary_report.py` | Integrated HTML report generator | prior summaries/CSV/SVG | `gait_summary_report_20260312/gait_speech_summary_report.html` |
| `make_sci_figures_svg.py` | Generate SCI-style gait figures | single-dual outputs | `figures_sci/Figure*.svg` |
| `clean_merged_gait_pred_pool.py` | Clean and deduplicate merged gait pool | merged gait CSV | `merged_gait_pred_2025_2026_*` |
| `rerun_full_pred_pool_ms04.py` | Rerun feature pool with min stride 0.4 settings | pred JSON pools | `pred_2025_rerun_ms04_features.csv`, merged rerun files |
| `rerun_pred2026_recovered_with_new_metadata.py` | Reprocess recovered pred2026 JSON with metadata | recovered JSON | rerun/recovered CSV |
| `rerun_suspicious_with_new_metadata.py` | Rerun suspicious rows with updated metadata | suspicious rows | rerun suspicious CSV |
| `rerun_suspicious_compare_min_stride.py` | Compare min-stride settings for suspicious rows | suspicious rows | ms comparison CSV/JSON |
| `analyze_event_detection_failures.py` | Diagnose event detection failures | gait features / logs | `event_detection_failure_diagnostics.*` |
| `simulate_event_detection_sweeps.py` | Sweep event detection settings | failure cases | `event_detection_sweep_summary.json` |
| `analyze_overlap_with_normalization.py` | Compare tele/pred overlap with raw and normalized deltas | overlap gait xlsx | `tele_pred2025_overlap_analysis/` |
| `analyze_overlap_keypoint_scale_normalized.py` | Keypoint-scale normalized tele vs pred overlap | overlap keypoints/xlsx | `tele_pred2025_keypoint_normalized_overlap/` |
| `compute_perspective_normalized_features.py` | Perspective-aware normalized gait features | keypoint JSON | perspective feature CSV |
| `compare_perspective_corrected_overlap.py` | Compare perspective-corrected overlap | perspective feature CSV | `tele_pred2025_perspective_correction_overlap/` |
| `apply_calibration_ratio_to_overlap.py` | Apply global calibration scale ratio to tele spatial metrics | calibration ratios + overlap CSV | `tele_pred2025_calibration_ratio_attempt/` |
| `analyze_new330_clean_gap.py` | Analyze gap between current gait pool and new330 data | merged clean CSV | `new330_clean_gap_*` |

## 4. Output Directory Inventory

| Directory | Meaning | Key files |
|---|---|---|
| `new_data_check_analysis/` | Demographic baseline and demo-only ML | `summary.json`, group and CV result CSVs |
| `single_dual_analysis/` | pred2025/pred2026 single-vs-dual deltas | `summary_all.json`, `pred_2025_*`, `pred_2026_*` |
| `gait_tele_single_dual_analysis/` | tele single-vs-dual deltas from xlsx | `summary.json`, `delta_stats.csv`, `pair_values.csv` |
| `matched_pred2025_gait_analysis/` | pred2025 gait matched to demo | `summary.json`, `matched_dataset.csv`, model CSVs |
| `matched_pred2025_gait_analysis_moca/` | pred2025 gait matched to demo, MoCA-focused variant | `summary.json`, model CSVs |
| `matched_pred2025_by_task_analysis/` | pred2025 task-split analysis | `single/`, `dual/`, `summary.json` |
| `matched_pred_pooled_by_task_analysis/` | pooled pred2025+pred2026 analysis | `single/`, `dual/`, `summary.json` |
| `matched_pred_pooled_by_task_analysis_fast/` | current report-facing pooled analysis | `single/`, `dual/`, `summary.json` |
| `dual_task_cost_gait_speech_analysis/` | older xlsx speech workflow | 15 rows, regression only |
| `dual_task_cost_gait_speech_analysis_task0809/` | old TASK08-09 xlsx speech branch | 14 rows, regression only |
| `dual_task_cost_gait_speech_analysis_task0809_compare_old0306/` | old 0306 speech xlsx comparison branch | 14 rows |
| `dual_task_cost_gait_speech_analysis_task0809_compare_new0421_endsec/` | newer 0421/endsec speech xlsx comparison branch | 14 rows |
| `dual_task_cost_gait_speech_json_analysis/` | current task8/task9 JSON gait-speech workflow | 115 rows, classification + regression |
| `gait_summary_report_20260312/` | integrated HTML summary | `gait_speech_summary_report.html` |
| `figures_sci/` | SVG figures for report/manuscript | `Figure1_effect_size_forest.svg`, etc. |
| `similarity_results/` | early gait similarity comparison | `gait_similarity_report*.md`, `gait_similarity_results.csv` |
| `tele_pred2025_*` | tele vs pred2025 overlap/calibration/normalization experiments | overlap and delta CSVs |

## 5. Current Best Interpretation

The current strongest gait-speech workflow is the JSON-based one:

```text
MMPose gait JSON
  -> gaitfeature.py
  -> check_previous_format_pred_2025_adaptive.xlsx / check_previous_format_new_2026_adaptive.xlsx
  -> legacy_exploratory/analyze_dual_task_cost_gait_speech_from_json.py
  -> dual_task_cost_gait_speech_json_analysis/
  -> build_gait_summary_report.py
  -> gait_summary_report_20260312/gait_speech_summary_report.html
```

Why this is the current main line:

- It uses 115 matched gait-speech-demographic rows, unlike the older xlsx speech workflow with only 14-15 rows.
- It supports both label classification and MoCA/MMSE regression.
- It is the source consumed by the integrated HTML report.
- It directly uses rerun `task8` and `task9` JSON speech features, which are more complete than the older speech xlsx branch.

Current analysis conclusion preserved in the report:

- Single-task gait is stronger than dual-task gait as a raw gait feature set.
- Speech dual-task cost is stronger than gait dual-task cost in the current joint dataset.
- Joint gait + speech has not yet outperformed speech-only, so the next useful iteration is feature pruning and direction normalization rather than adding more raw features.

## 6. Known Caveats

- Several workflows depend on files outside this folder, especially `../new data_check.xlsx` and speech artifacts under `server_run_artifacts_20260311_89/`.
- Some scripts use `pandas`; the default local `python3` checked during inventory did not have `pandas` installed.
- Older speech-xlsx workflows are retained for comparison but are underpowered because all matched rows are `label=1`.
- Camera calibration experiments are useful QC branches but are not part of the current best gait-speech modeling pipeline.
- Cost direction is currently `(dual - single) / single`; for clinical interpretability, a later pass should normalize feature direction so positive consistently means worse.

## 7. Recognition and Feature Extraction Code Availability

### Gait

Available locally:

- `gaitfeature.py`: current gait feature extraction code. It consumes MMPose keypoint JSON/video metadata and outputs gait metrics JSON.
- `gait_features_extraction_without_ds_new.py`: legacy non-dual gait feature extraction code.
- `gait_features_extraction_ds.py`: legacy dual-task phase feature extraction code.

Not fully available locally:

- The actual MMPose video-to-keypoint recognition batch script is not present in this local `gait` workspace.
- Local evidence exists in `pred2026_consistency_rootcause.md` and `pred_2026/process_log.txt`, including MMPose model choices such as `human/rtmpose-m` and `td-hm_hrnet-w48`.
- Server-side scripts mentioned in prior work include `batch_mmpose.sh`, `mmpose_2025.sh`, and `mmpose_260226.sh`, but they are not currently stored as local files in this workspace.

### Speech

Available in sibling folders, not inside `gait/`:

- `../feature_extraction_code/optimization_schemes/server_asr_recognize.py`: WhisperX ASR recognition script.
- `../feature_extraction_code/optimization_schemes/server_asr_recognize_opt.py`: optimized WhisperX ASR script with task-specific presets.
- `../feature_extraction_code/optimization_schemes/server_extract_features.py`: Serial-7 speech feature extraction from ASR JSON.
- `../feature_extraction_code/optimization_schemes/server_extract_features_opt.py`: optimized Serial-7 feature extraction with Chinese-number and compact-digit handling.
- `../feature_extraction_code/optimization_schemes/01_preprocess_audio.py`: audio preprocessing.
- `../feature_extraction_code/optimization_schemes/02_rule_constrained_postprocess.py`: rule-constrained Serial-7 postprocessing.
- `../feature_extraction_code/optimization_schemes/run_all_pipeline.sh`: speech pipeline runner.
- `../feature_extraction_code/optimization_schemes/run_server_pipeline_opt4.sh`: server speech pipeline runner.

Also available as archived server artifacts:

- `../server_run_artifacts_20260228/code/server_asr_recognize.py`
- `../server_run_artifacts_20260228/code/server_extract_features.py`
- `../server_run_artifacts_20260228/asr_json_v2/`
- `../server_run_artifacts_20260228/features_*.json`
- `../server_run_artifacts_20260311_89/features_v2_8_0306.json`
- `../server_run_artifacts_20260311_89/features_v2_9_0306.json`

Current gait-speech analysis inputs copied into `gait/`:

- `task8_features_rerun_20260311.json`
- `task9_features_rerun_20260311.json`

These two JSON files are speech feature outputs used by `legacy_exploratory/analyze_dual_task_cost_gait_speech_from_json.py`; they are not the ASR recognition code themselves.
