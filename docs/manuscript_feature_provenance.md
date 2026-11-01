# ADualNet2606 feature provenance

## Scope

Historical modeling from a saved participant-level table and extraction from raw
recordings are separate reproducibility claims. The current extraction utilities
have not been certified to regenerate the historical 164-person table. Matching
column names does not make the measurement definitions interchangeable.

The 2026-10-30 audit confirmed that all 2460 gait cells in `dataset_2606.xlsx`
match `check_previous.xlsx`. All 4428 cells in the saved primary modeling snapshot
(ID, 25 predictors and outcome) match the original table. Subsequent author-approved
speech corrections belong to a separate data version and have not been evaluated.
The published-draft mean-fold AUROC 0.893 remains bound to the original snapshot.

## Reconstructed historical gait definitions

The archived non-phase extractor calculates image-coordinate displacements.
It divides these values by a nominal route-average speed to produce four columns
named step/stride time and variability. Those ratios are not physically measured
seconds. Its `cadence` equals 60 divided by this legacy stride ratio, not a validated
steps-per-minute measurement. A guessed global scaling factor or simply doubling
cadence does not repair the dimensional inconsistency.

Of 154 exact-ID saved keypoint files, 152 reproduce all 11 archived non-speed
outputs using the saved speed as an input. These saved files contain 17 keypoints
per instance, not 15. Candidate alternative filenames reproduce the spatial
formulas for the remaining 12 records, but numerical agreement is not source-identity
certification. One record has different candidate files matching its spatial and
phase fields. No sources were automatically merged or replaced.

The author confirms that all subjects originally used a common set of phase
thresholds. The exact values and historical cache/formula/FPS versions remain
unreconciled. With the currently archived phase formula, FPS 30 and the 152 direct
spatial-matching caches, no single velocity threshold reproduces all saved stance
durations. This rejects that particular reconstruction, not the author's statement,
and does not prove diagnosis-dependent preprocessing. Preserve the group argument
in archived code for provenance; never choose a new subject's settings from its
diagnosis or outcome.

Phase outputs are accumulated counts of classified frames divided by FPS. They
must not be described as per-step mean stance, swing or confirmed foot-contact
durations without the original definition and reference validation.

Read-only server1 verification found 140 exact-ID old videos. All 140 saved
speeds equal 6 m divided by full-clip duration rounded half-up to 0.01 seconds;
all 140 video streams report 30/1 FPS. This supports a nominal full-task/clip
speed, not steady walking-only speed. The remaining 24 source correspondences
and the image-coordinate spatial calibration are not thereby certified.

The server's `postprocess_gait_res_2026.sh` explicitly uses common HC settings
(velocity 1.089, acceleration 1295, FPS 30). It targets a 2026 reprocessing
directory and has not been linked to the original 164-person feature table.
Checked old-result paths and relevant bash-history entries did not recover the
original run. Do not insert these later defaults into the manuscript as if
the original settings had been established.

## Speech provenance

The author confirms manual transcript checking and correction before feature
calculation. The 5.41% number-token error rate describes uncorrected ASR on 29
samples. It is not the error rate of the corrected inputs.

The current speech extractor includes expected-sequence snapping, count floors
and latency offsets. Many historical rows violate those output fingerprints;
therefore it is not a certified historical extractor. Do not apply its corrections
to manually verified participant errors or assume identical scoring semantics.
Sequential-minus-seven vs fixed-target correctness, response inclusion, timestamp
anchors and zero-response handling still need annotation-level documentation.
Rows 132 and 161 retain unresolved first-20-second/full-task count differences.

## Read-only check

`adualnet_2606_feature_contract.json` records every column, role, formula, unit
boundary and required action without participant identifiers. Run:

```bash
python scripts/audit_manuscript_feature_table.py \
  --input-table /path/to/dataset_2606.xlsx --sheet info \
  --report-json /path/to/feature_audit.json \
  --require-publication-ready
```

The audit never fits or scores a model and never changes the input workbook.
Exit 2 means the recorded publication blockers remain. A changed file or renamed
column does not clear them. This check is separate from historical training
entrypoints; it does not silently alter their inputs, results or parameters.

Before a new analysis, approve a versioned measurement definition, document
source identities and common extraction settings, resolve annotation discrepancies,
and freeze the resulting input contract. Do not mix remeasured columns with old
checkpoints or report old scores as corrected-data results.
