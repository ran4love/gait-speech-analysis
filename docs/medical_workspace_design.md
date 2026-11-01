# ADualNet medical research workspace

Version 0.3.0, prepared 2026-11-01.

## Design basis

The owner requested a layout suitable for medical researchers, referring to
https://neurovision-app.vercel.app/ and subsequently specifying familiar
black, navy and white backgrounds. The reference URL timed out in browser
inspection and could not be fetched reliably. The owner subsequently supplied a
screenshot and clarified that colors and displayed content, not structural
imitation, were the priority. The screenshot informs the navy/white palette,
clear summaries and module explanations. ADualNet retains its own task-oriented
navigation and review layout; the reference's branding, diagnosis claims,
accuracy figures and patient counts are not copied.
All navigation icons are inline code-native SVG; no reference-site assets,
external fonts or network-delivered UI libraries are used.

## Information hierarchy

- Persistent navy navigation: video, synthetic modelling demo, cohort table, advanced workflows.
- Compact white top bar: research context, research-only boundary and language switch.
- Task context: TUG-DT, recognition engines, dual human confirmation and actual export formats.
- Truthful summary cards: current task state, gait/speech modalities, review requirement and export formats.
- Four-step workflow: upload, recognition, human review, features/QC.
- Recording and recognition-environment cards: input handling distinct from technical setup.
- Review area: synchronized pose/video and per-response annotations; raw recognition remains immutable.
- Collapsed original logs and expandable usage/data-handling explanation.
- Gait, speech and cohort-modelling capability cards, with measurement and interpretation boundaries.

No invented subject counts, model accuracy scores, connection-ready indicators,
certification badges, automatic diagnosis or cloud-storage promises are displayed.
Completed status refers to software output only. A synthetic job is labelled as
synthetic in the review area. Server availability is checked by the explicit
environment button, not inferred from the visual theme.

## Interaction boundaries

Workflow status comes from existing video jobs and local unsaved-review state.
It is a coarse task stage, not a percent-complete or clinical-QC score. The review
shortcut is enabled only once reviewable job data are ready. Editing retains the
existing confirmation-clearing behavior and updates only its presentation status.
Changing tabs or language does not save, approve or export annotations. Unknown
runtime status defaults to an uncompleted state; failures are not shown as success.

Both human confirmations, per-response correctness, source retention and versioned
exports remain required. Recognition/model settings and measurement definitions
are not changed. Existing figures, historical results and frozen PD models/cohorts
are outside the scope of this interface change. No clinical recordings are needed
for UI acceptance testing, and no real-data classifier inference is authorized here.

## Accessibility and responsive use

Buttons retain text labels alongside icons. Keyboard focus is visible; workflow
position uses aria-current and the state text does not rely solely on color.
Reduced-motion preference is respected. Narrow layouts move navigation above
the content and stack the operation/review columns; response tables scroll within
their own container rather than widening the entire page. English and Chinese
labels, placeholders and status summaries remain supported. Native media controls
and OS file dialogs retain their browser/OS language.

The loopback server and optional configured Ubuntu recognition environment remain
unchanged in purpose. This visual redesign does not certify usability, accessibility
conformance, clinical accuracy, security compliance or deployment readiness for a
multi-user hospital system. Institution-specific evaluation is still required.
