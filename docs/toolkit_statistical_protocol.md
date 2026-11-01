# ADualNet toolkit 0.1 statistical protocol

## Scope and input

This prospective, feature-table research tool evaluates full-cohort within-task
TUG-DT domain ablation. It does not ship a clinical classifier or authorize
historical data corrections. Only the synthetic demonstration was executed while
preparing this release. Existing manuscript and PD experiments remain archived.

The toolkit requires one record per participant and rejects duplicate IDs,
missing/invalid binary labels, ambiguous aliases, missing candidate columns,
entirely missing features, infinities and nonnumeric cells. Outcomes must be
exactly 0 and 1 before any integer conversion. Extra columns, including cognitive
scores and clinical categories, are excluded from predictors and listed in
preflight. Sex is a numeric coded variable; supply its codebook. Feature definitions
and physical units are the study owner's responsibility.

## Models and validation

Reuse `make_publication_classification_models`, `safe_fit` and
`select_top_k_features_train_only` from the historical reproduction code.
AdaBoost, RF, SVM (RBF) and DT retain those model parameters. Median imputation,
scaling where applicable and feature ranking use only outer training data.
Ranking uses absolute distance of univariate training AUROC from 0.5, selecting
min(14, available candidate count). There is no hyperparameter search in this
toolkit; 14 is a fixed cap, not a newly tuned optimum.

The default uses RepeatedStratifiedKFold, 5 folds, 10 repeats, seed 42. Every
feature set uses the same complete cohort and identical split indices. Each
participant has exactly one held-out probability per repeat per model/feature
set. A fixed probability threshold of 0.5 is used. No evaluation-label-dependent
threshold selection is performed. Feature selection is nested within training
folds; this should not be confused with a separate inner tuning loop.

Compare demographics, gait, speech, gait+speech and gait+speech+demographics.
Additional demographic subsets are education, age+sex and age+education. The
three-variable demographics set already supplies age+sex+education.

## Point estimates and SD

For each repeat, calculate AUROC, accuracy, precision, recall, F1 and Brier on
the complete OOF vector for all participants. Report the arithmetic mean across
repeats; report sample SD across repeat estimates. Repeats reuse the same cohort
and are not independent biological samples. Fold-level AUROC and complete-repeat
metrics are retained separately. Pooled repeated prediction ROC is descriptive.

The historical manuscript pipeline reported mean-fold performance. These
estimands can differ even with identical predictions; a different number must
not be represented as a correction to historical performance without an explicit
reanalysis and new data/result version.

## Conditional 95% intervals and paired differences

Default 2,000 bootstrap draws, seed 20261031. Sample participant IDs with
replacement separately within outcome classes, retaining observed class counts.
All repeated held-out predictions travel with a sampled participant, including
multiple copies if that ID is sampled more than once. Recalculate each repeat's
metric under the resulting participant weights, then average across repeats.
Use empirical 2.5th and 97.5th percentiles. Shared weights for every model and
feature set preserve pairing. Paired AUROC intervals use percentiles of the
within-draw difference, not differences between two interval endpoints.

These intervals are conditional on the cached CV fits and observed class counts.
They handle repeated prediction rows at the participant level, but do not fully
capture dependence induced by overlapping training cohorts or uncertainty from
refitting, feature selection and tuning. They are not guaranteed nominal-coverage
intervals for the complete training procedure or external population performance.
Stratification also conditions precision/accuracy-related results on the observed
class composition. Small samples and few repeats require particular caution.

The historical fold bootstrap remains available for historical reproduction:
it resamples correlated fold metrics as separate observations and does not
account for participant reuse/shared training dependence. Its nominal 95% label
does not establish population-level 95% coverage. A prose clarification alone
does not repair that inferential limitation. Historical outputs were not changed.

## Calibration and demographics

Export reliability bins per repeat, with participant count in each nonempty bin;
use 10 equal-width probability bins. Display pooled repeated probabilities as a
descriptive curve. Brier and ECE are calculated per repeat and averaged. The
conditional bootstrap performance table includes Brier intervals; no binwise
interval, calibration intercept/slope or recalibration model is estimated.
Discrimination alone does not establish good probability calibration.

Covariate addition/removal evaluates predictive contributions, not causal
adjustment or education-independent biomarker validity. Education-only and
age+sex comparisons help interpret demographic structure. All-three recurrence
is tautological when three of three features are selected. Education strata use
the prespecified <12 vs >=12 years cutoff on full-cohort OOF predictions; they
are descriptive subsets, not separately trained or externally validated cohorts.
No subgroup AUROC is estimated if either class has fewer than five subjects.

## Audit outputs

`used_config.json` and `protocol.json` record version, input SHA256, seeds,
dependencies, measurement/outcome definitions, estimand and limitations.
`feature_mapping.csv`, missingness, anonymous splits, selected features, per-fold
AUROC, per-repeat metrics, anonymous predictions and confusion matrices are
retained locally. Reports and XLSX tables present aggregate results.

Optional SHAP and regression scripts remain in the historical workflow. The
browser toolkit does not relabel fixed-panel historical SHAP/regression as fully
nested analyses; compare their data and selection scope before manuscript use.
