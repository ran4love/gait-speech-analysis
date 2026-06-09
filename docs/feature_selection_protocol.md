# Feature Selection Protocol and Leakage Audit

This note documents how feature selection is handled in the public reproduction code.

## Nested-Selected Features

The default manuscript panel is now `Nested-selected`, a fixed 14-feature panel
derived from the nested feature-selection frequency analysis. The panel contains:

```text
ACC_DT, Correct_Num_20s_DT, Correct_Num_DT, Correct_persec_DT, Num_20s_DT,
Num_DT, Timecount_DT_1stNum, age, edu, swing_times,
average_step_assymetry, average_stride_assymetry, stance_times,
average_stride_length
```

This fixed panel is used by default for the main repeated-CV benchmark,
confounder adjustment, regression, and the publication table. It is derived from
training-fold rankings, but once frozen it should be described as a
`nested-frequency-derived fixed panel`.

Leakage interpretation:

- The strict leakage-controlled result remains the nested feature-selection CV
  output, where feature ranking and top-k selection are repeated inside each
  training fold.
- The fixed `Nested-selected` panel is useful for a stable manuscript feature
  set and downstream SHAP/regression analyses.
- Do not describe the fixed-panel repeated-CV result as identical to the nested
  feature-selection result; report both if space permits.

## Historical ADual-Selected Features

The original historical `ADual-selected features` are still available through
`--feature-set adualselected` for backward-compatible re-analysis:

```text
Timecount_DT, average_step_assymetry, stance_times, swing_times, walking_speed,
Num_DT, Correct_Num_DT, Num_20s_DT, Correct_Num_20s_DT, Correct_persec_DT,
ACC_DT, Timecount_DT_1stNum, edu, age
```

It is not the default because it was not selected inside the current nested CV
protocol.

## Nested Feature Selection Output

To provide a leakage-controlled feature-selection analysis, the public code
includes an explicit nested feature-selection workflow:

```bash
python run_pipeline.py --config config.ini --step paper-experiments
```

with these config values:

```ini
run_nested_feature_selection = yes
nested_candidate_set = script_all
nested_select_k = 14
feature_set = nestedselected
promote_nested_selection = yes
```

With `promote_nested_selection = yes`, the main classification outputs
(`model_performance_table`, ROC, calibration and confusion matrix) are copied
from nested held-out predictions. Fixed-panel publication outputs are retained
as `fixed_panel_*` files for audit and sensitivity checks.

The nested output is written under:

```text
results/paper_experiments/classification/nested_feature_selection/
```

Key output files:

| File | Purpose |
| --- | --- |
| `nested_feature_selection_protocol.json` | Machine-readable protocol statement. |
| `nested_feature_selection_performance.csv/.xlsx` | Mean, SD, and bootstrap CI performance after nested selection. |
| `nested_feature_selection_fold_metrics.csv` | Per-fold held-out metrics. |
| `nested_selected_features_by_fold.csv` | Selected features in each outer fold. |
| `nested_feature_rankings_by_fold.csv` | Training-only feature ranking in each outer fold. |
| `nested_selected_feature_frequency.csv` | How often each candidate feature was selected. |
| `nested_feature_selection_predictions.csv` | Held-out predictions only. |

## How Nested Selection Is Performed

For each repeated 5-fold CV outer fold:

1. Split data into training fold and held-out fold.
2. Use only the training fold to rank candidate features.
3. Ranking score is `abs(training_univariate_AUROC - 0.5)`.
4. Select the fixed top `nested_select_k` features from the training fold ranking.
5. Fit imputation, scaling where applicable, and the classifier using only the
   selected training-fold columns.
6. Predict the held-out fold once.
7. Store fold metrics, predictions, selected features, and training-only rankings.

The held-out fold is never used for feature ranking, top-k selection, imputation
fitting, scaling fitting, or model fitting.

## Top-k Tuning Rule

`nested_select_k` is a fixed protocol parameter. If users want to tune `k`, it
must be tuned either before the final evaluation on an independent discovery set
or inside an inner CV loop nested within each outer training fold. Do not choose
`k` based on outer-fold test performance.

## Older Exploratory Scripts

The following scripts contain forward-selection utilities and are retained as
exploratory analysis helpers:

```text
src/legacy_exploratory/analyze_dual_task_cost_gait_speech_from_json.py
src/legacy_exploratory/analyze_matched_pred2025_gait_vs_demo.py
src/legacy_exploratory/analyze_matched_pred2025_by_task.py
```

Their forward-selection outputs should not be used as final unbiased model
performance unless the feature-selection step is nested inside the validation
loop. For paper-level public reproduction, use
`src/paper_experiment_reproduction.py` and the nested feature-selection outputs
listed above.
