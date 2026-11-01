# Public editor demonstration

## Online entrypoint

https://ran4love.github.io/gait-speech-analysis/

The site is published by `.github/workflows/pages.yml` from the **public_demo/**
directory only. The complete research toolkit is in the GitHub repository;
local workspaces, real recordings, participant tables, credentials, model weights
and generated study results are not part of either publication.

## What editors can do without installation

1. Open the mathematical synthetic example and inspect its pose animation/timeline.
2. Read the scripted response sequence, edit/include/exclude response rows and mark correctness manually.
3. Adjust speech task boundaries and record a fictional reviewer code and explicit scoring rule.
4. Confirm both gait/reference and speech review, then generate feature/source/QC outputs.
5. Download an XLSX workbook, feature CSV, provenance JSON or an unconfirmed review-draft JSON.

The reference example has four scripted responses: 93, 86, 78, 71. An example
sequential subtraction convention retains the error 78 rather than snapping to
79. If 71 is judged relative to the preceding 78, three responses are correct,
accuracy is 0.75 and correct responses per second is 3/12. Correctness is never
filled automatically. Changes clear both confirmations and hide old downloads
until the revised annotations are confirmed again.

## Deliberate scientific limits

No online neural inference or classifier fitting/scoring occurs. The animation
is not recorded human video/audio; the transcript is scripted, not ASR output.
Gait feature values are precomputed on a fixed 0–12s synthetic pose fixture with
the repository's unchanged `gaitfeature.calculate_gait_parameters` function.
The fixture records its extractor hash, pose hash, outputs and actual QC flags.
The mathematical excursions exercise the existing image-space event detector;
their amplitudes are illustrative, not physiological ground truth or an accuracy test.
Failed QC is not replaced by a fabricated pass. This public demo does not support
gait-window or keypoint edits that would require recomputing gait references.

Live browser calculations use the reviewed response rows. Their eight speech
definitions match `reviewed_response_onset_counts_v1`, with [start,end) task
windows and an exclusive first-20-second upper boundary. Zero-response accuracy
and latency remain missing. There is no minimum count, timing offset or expected
sequence correction. Free-text transcript edits are audit notes, not response counts.

Contact-phase metrics remain missing. Pixel-space displacement is not metric
step length, and step=stride/2 is an algorithm constraint rather than independent
physiological validation. No clinical performance claim can be inferred from
this software demonstration or its synthetic results. New measurement versions
are not certified reproductions of the original manuscript feature table.

Real MMPose/Whisper inference, video review and gait extraction remain in the
local/authorized-server toolkit. Cohort modelling requires a separate input table
and protocol. Do not use the demo as a diagnosis or as remote patient processing.

## Privacy and exports

The public site has no upload controls, application API calls, analytics scripts,
external libraries, fonts, GPU access or credentials. Its Content Security Policy
blocks application connections (`connect-src 'none'`) and form submissions.
Review edits reside only in tab memory; reload clears them. Only the language
preference may be saved in local browser storage. Never enter real patient data.
GitHub may log ordinary website requests/IP addresses under its own policies.

Downloads are generated as browser Blob objects, not uploaded. The dependency-free
XLSX writer uses inline strings, not formulas/macros; CSV cells are escaped against
formula injection. Inspect downloaded notes before sharing them. Workbooks separate
features, definitions/sources, reviewed responses, QC and provenance. Synthetic
reference gait values and live speech values are explicitly distinguished.

## Rebuild, test and publish

```bash
python scripts/build_public_demo_fixture.py
python scripts/build_public_demo_fixture.py --check
python -m unittest discover -s tests -v
node tests/public_demo_checks.js /tmp/adualnet-demo-check.xlsx
python -m http.server 8876 --bind 127.0.0.1 --directory public_demo
```

The generator only creates mathematical coordinates and scripted responses; it
never reads a cohort table or local workspace. Do not replace its input with
participant media. Pages CI tests privacy boundaries and fixture provenance,
then publishes only public_demo. Local Python dependencies are not deployed
to Pages. MIT applies to the original source; third-party components retain
their own licenses and are not embedded in this demo.
