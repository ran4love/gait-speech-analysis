# ADualNet 0.4.1

Date: 2026-11-02

- Prevent response-action buttons from wrapping character-by-character in narrow review columns; overflow remains within the response table.
- Version static assets to avoid stale browser caches after release updates.
- No changes to inference, measurement definitions, feature computation or statistical analysis. The v0.4.0 release is preserved.

# ADualNet 0.4.0

## Public synthetic demonstration (2026-11-01)

Added an independent, static browser demo for readers/editors: pose animation,
manual response review, live reviewed-speech calculations and browser-only XLSX,
CSV and JSON downloads. Gait references are explicitly precomputed from fixed
mathematical poses; no online recognition or classifier is simulated as real.
There is no real-data upload, application API, external UI library or telemetry.
The public artifact directory is separated from the full local toolkit and
audited by tests. GitHub Pages publication is defined by a dedicated workflow.

The owner selected MIT on 2026-11-01. LICENSE and LICENSE_NOTICE.md supersede the
earlier pending-license state for original ADualNet source. Historical release
notes below retain their original preparation status. Third-party terms remain.

See public_demo.md for exact scope, privacy, reference/live feature distinctions
and how to rebuild the synthetic fixture. Existing measurements, study outputs
and frozen PD models/cohorts are unchanged.

## Medical research workspace (2026-11-01)

Navy navigation, white operation surfaces, compact headings and grouped recording /
environment cards replace the presentation-led landing page. A four-step indicator
reflects coarse video-job state, not a completion percentage or physiological QC
certification. The review shortcut is unavailable before review data exist; editing
a reviewed case returns the presentation indicator to human-review-needed.
Technical logs remain available in a collapsed section. English/Chinese support,
manual gates and immutable exports are preserved. No external fonts, analytics,
reference-site assets or cloud publishing were added.

33 unit/integration checks pass. Browser acceptance used synthetic data only:
recognition-demo processing, gated review navigation, rejection before manual
confirmation, bilingual annotations, saved review and XLSX/CSV/QC export all
completed. The reviewed fixture retained four responses, three correct and
ACC=0.75. English and Chinese 390px layouts had no whole-page horizontal overflow;
no application console errors were observed. The gait extractor, recognition
worker, SSH adapter, video feature implementation, reviewed-speech definitions,
modelling, uncertainty and report code match the prior 0.2.1 source archive.

The supplied NeuroVision URL could not be reliably retrieved in this environment.
The owner then supplied a screenshot and prioritized its colors/content rather
than structural imitation. The final palette follows that navy/black/white
preference while retaining ADualNet's own workflow. No reference-site accuracy
figures, diagnostic claims or patient counts are copied. This is not a diagnostic
or hospital-certified clinical interface. See medical_workspace_design.md.

The following sections record earlier release verification.

Prepared 2026-10-31. Source candidate only; license approval and GitHub publication
remain with the owner. No participant data or pretrained classifier is included.

## Bilingual interface in 0.2.1

The English / 中文 header controls switch labels, placeholders, review controls,
progress summaries, validation summaries and accessibility labels in place.
English is the initial default and the selection is remembered in this browser.
Changing language does not reload, clear unsaved edits, reset video playback,
change selected tracks or alter review confirmations. Transcript content, response
text, IDs, filenames, scoring rules, notes, raw logs and exports are not translated;
existing HTML reports retain their original language. Native OS dialogs and video
player controls follow the browser/OS locale. File-selection buttons inside the
application are bilingual.

31 tests pass, including translation completeness, matching placeholders, duplicate
catalog keys, backend progress coverage, authenticated asset delivery and guards
against translating user-authored data. Browser checks verified English/Chinese
switching, unsaved review-state preservation, language preference persistence,
localized input errors, synthetic feature-table preflight and 390px layouts.
No recognition/feature formulas or model settings changed in this translation patch;
GPU recognition was not rerun. The following documents prior 0.2.0 verification.

## Single-video workflow added in 0.2

- Local or Ubuntu SSH MMPose + Whisper adapters, with separate recognition Python environments.
- FFmpeg normalized video/audio timeline, raw output retention, resolved pose/detector configs and package provenance; hashes for explicitly configured pose/detector weights.
- Video playback and skeleton overlay, frame stepping, participant-track selection, per-frame track/point corrections and exclusion without elapsed-time collapse.
- Raw/corrected transcript, response rows, per-answer manual correctness and recorded scoring rule.
- Two manual confirmations required before versioned XLSX/CSV/QC export; immutable reviews and raw recognition artifacts.
- SSH transfer acknowledgement, no password fields, loopback-only UI and allow-listed seekable media routes.
- Public SSH template and Chinese video walkthrough; local server settings and all media remain outside the source-only release.

## Verification and boundaries

The review demonstration completed draft saving, explicit gait/speech confirmation
and XLSX/CSV export in the browser; an unconfirmed attempt was rejected. The demo
is artificial, not a neural inference or clinical accuracy test. The browser
console showed no errors. A public MMPose repository photograph was separately
encoded into a three-second silent video and processed on Ubuntu/RTX 3090 using
the existing MMPose environment and faster-whisper large-v2 environment.
MMPose returned all 90 frames, and Whisper correctly returned no speech segments.
This verifies transport/adapter execution only, not gait-event accuracy, ASR
accuracy on clinical speech, patient tracking or physiological measurement validity.

An actual RTMPose SimCC output exceeded 1; the adapter was corrected to preserve
finite nonnegative raw scores rather than treating them as probabilities.
All 26 tests passed in the working source and in a relocated source archive.
Tests cover that case, processing authorization, per-response scoring, version
gates, raw preservation, stale reviews, timeline integrity, coordinate bounds
and media ranges alongside the existing toolkit/statistical tests. A detector
weight/config-pair guard prevents MMPose defaults from silently ignoring requested
weights. The final server smoke test used explicit local detector and pose
weights; both adapters completed and the job entered awaiting_review.

The gait measurements remain experimental image-space ankle-event estimates.
Step=stride/2 is imposed, not independent evidence of physical accuracy. Three
contact-phase fields remain explicitly unavailable; no diagnosis-dependent phase
presets or fabricated values are used. No video export is automatically scored
by a classifier. Existing manuscript and PD results/models are untouched.

In SSH mode both machines retain media and transcripts, including failed-job
artifacts. This is a single-researcher workbench, not a multi-user clinical service.
Installed recognition environments/weights must be independently pinned and
validated by a new site. Optional WhisperX alignment is not certified by the
faster-whisper smoke test. See video_workflow_zh.md for the full limitations.

## Historical 0.1 verification

The following describes the earlier feature-table release preparation only.

Prepared 2026-10-31 for researchers trying the gait-speech TUG-DT paradigm.
License approval is pending. This source candidate has not been pushed to GitHub.

## Added

- `adualnet.py`: serve, demo, validate, analyze, reviewed-speech and pipeline commands.
- Browser interface with local uploads, preflight, progress and HTML/XLSX reports.
- Shared-cohort repeated CV with training-fold ranking and original model constructors.
- Conditional participant cluster bootstrap, paired differences and explicit estimand.
- Calibration/Brier/ECE, demographic component models and education strata in reports.
- Human-reviewed speech response counts with an explicit, separately versioned definition.
- Blank templates, fully synthetic example, Chinese guide and statistical protocol.
- Source-only release builder, file hashes, test suite and GitHub Actions definition.

## Verified locally

Synthetic CPU CLI and browser demonstrations completed all 192 model fits
(8 feature sets, 4 models, 3 folds and 2 repeats). Browser demonstration used
200 bootstrap draws. Output reports, source tables and XLSX were generated.
The browser upload/preflight flow passed with the 80-subject synthetic CSV;
browser console had no errors. Desktop and 390-pixel mobile layouts were viewed.

Tests cover input validation, ambiguity and duplicated identifiers, fractional
labels, missingness, weighted AUC ties, repeated-prediction invariance, paired
differences, response annotation semantics and split/output invariants. Existing
historical feature-audit tests are included. Source compilation and diff checks
passed. CI is supplied for Python 3.11/3.12; hosted CI has not been executed here.
Local checks used Python 3.14 and the installed numeric libraries.

## Boundaries

This preparation ran synthetic inputs only. Raw MMPose/WhisperX recognition was
not executed or GPU-tested; existing wrappers remain configurable. Existing
measurement provenance qualifications remain in force. The new reviewed-speech
definition is not a certified reproduction of historical manuscript annotation.

Historical manuscript outputs and PD models/cohorts/validation records were not
modified. New toolkit CIs remain conditional on cached fits; they do not supply
full training-procedure uncertainty. Historical fold bootstrap limitations are
documented without retroactively changing historical figures or claims.

Optional SHAP, regression and historical statistical scripts remain advanced
entrypoints; they are not automatically presented as new nested browser results.

Public release needs the owner's license decision, publication citation details
when available and a recognition-environment smoke test if raw inference will be
advertised as tested on a specific GPU/OS combination.
