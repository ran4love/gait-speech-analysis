# Measurement profiles and research claims

Column names alone do not establish equivalent measurements. Four profiles are
documented in workbench_measurement_profiles.json:

| Profile | Panel | Intended role |
|---|---|---|
| toolkit_table_v03 |15 gait,8 speech,3 demographics | Generic cohort-table interface; study-specific definitions required |
| video_review_v1 |15 gait fields, including3 structurally unavailable contact phases;8 speech | Reviewed recording export, not classifier-ready |
| historical_2606_primary |15 gait,8 speech,2 demographics | Historical25-predictor primary panel with provenance limitations |
| temporal_audit_20261031 |7 gait,6 speech,2 demographics | Separately registered measurement audit; no new workbench model entrypoint |

The historical domain analysis additionally considered gender. This does not make
it part of the historical primary25-predictor model.

## Read-only preflight

Run from the toolkit directory:

    python adualnet.py contract-check --input my_export.csv --profile video_review_v1

Use the relevant named profile. Add --out a_new_report.json to save a report;
existing files are never overwritten. This command uses only the Python standard
library and does not import, fit or score classifiers. It does not replace the
existing cohort numerical validator and currently accepts CSV only.

For a reviewed video, missing gait estimates can be legitimate QC outcomes.
The three undefined contact-phase fields must remain missing, never0.
The review exporter does not create demographic variables or an outcome label.
It must not be silently promoted to the26-predictor cohort interface.

The checker rejects mixed known measurement-version strings, missing/duplicated
identifiers, malformed numerics, entirely unavailable cohort fields and a known
video export presented as the cohort profile. A schema pass is not authorization
to model or evidence of source identity, physical accuracy or clinical validity.
Unknown study-specific definitions still need an explicit protocol; the checker
does not infer or certify their equivalence.

## Scientific boundaries

- Video spatial estimates are image pixels, not calibrated metric step lengths.
- Step=stride/2 is imposed by the current extractor, not independent validation.
- Ankle-event proxies are not certified foot contacts.
- Reviewed speech uses manual correctness labels and [start,end) task windows.
- Video zero-response accuracy/latency are missing. The separate temporal audit
  used accuracy0 for no response, so renaming columns would change semantics.
- Synthetic demonstrations verify execution and invariants, not patient accuracy.
- Historical classifier performance must remain attached to its actual
  measurement panel, sample, splits and statistical estimand.
- The workbench mean-complete-repeat OOF estimand differs from both historical
  mean-fold AUROC and the separate audit's participant-mean-probability AUROC.

The new command is additive. It does not change current formulas, model settings,
default feature sets, GUI routes or completed study outputs. Integrating a new
measurement panel into modelling requires a separately versioned implementation,
verification and study authorization.
