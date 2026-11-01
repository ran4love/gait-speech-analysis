"""Aggregate-only, self-contained browser reports and reusable source tables."""

import base64
import html
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_curve

from .uncertainty import bootstrap_weights, summarize_predictions

FULL = "gait + speech + demographics"
COLORS = {"AdaBoost": "#bd624c", "RF": "#367c8d", "SVM (RBF)": "#b58d3c", "DT": "#739071"}
CI_NOTE = ("95% intervals use outcome-stratified participant bootstrap of cached held-out predictions. "
           "All repeated predictions stay with the sampled participant; draws are paired across comparisons. "
           "Models are not refitted. These conditional intervals do not fully capture shared training-set "
           "dependence or uncertainty from refitting/selection. SD describes variation across repeats on the same cohort.")


def table(df):
    return df.to_html(index=False, border=0, float_format=lambda v: f"{v:.3f}", escape=True)


def reliability_rows(matrices, y):
    rows = []
    for (feature_set, model), probs in matrices.items():
        for repeat, p in enumerate(probs, start=1):
            bins = np.searchsorted(np.linspace(0, 1, 11)[1:-1], p)
            for b in range(10):
                mask = bins == b
                if mask.any():
                    rows.append({"feature_set": feature_set, "model": model, "repeat": repeat, "bin": b,
                                 "n_subjects": int(mask.sum()), "mean_probability": float(p[mask].mean()),
                                 "fraction_positive": float(y[mask].mean())})
    return pd.DataFrame(rows)


def education_groups(out, matrices, y, education, bootstrap_count):
    rows = []
    for name, mask in [("Education <12 years", education < 12), ("Education >=12 years", education >= 12)]:
        yy = y[mask]
        counts = np.bincount(yy, minlength=2)
        if counts.min() < 5:
            rows.append({"education_group": name, "n_subjects": int(mask.sum()), "n_label_0": int(counts[0]),
                         "n_label_1": int(counts[1]), "status": "not estimated: fewer than five participants in one class"})
            continue
        weights = bootstrap_weights(yy, bootstrap_count, 20261031)
        for (feature_set, model), probs in matrices.items():
            if feature_set not in [FULL, "gait + speech", "demographics-only"]:
                continue
            summary, _, _ = summarize_predictions(yy, probs[:, mask], weights)
            rows.append({"education_group": name, "feature_set": feature_set, "model": model,
                         "n_subjects": int(mask.sum()), "n_label_0": int(counts[0]), "n_label_1": int(counts[1]),
                         "status": "descriptive subgroup of full-cohort OOF predictions", **summary})
    result = pd.DataFrame(rows)
    result.to_csv(out / "education_stratified.csv", index=False)
    return result


def make_overview(out, summary, matrices, y):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    s = summary[summary.feature_set.eq(FULL)].reset_index(drop=True)
    for i, row in s.iterrows():
        axes[0].hlines(i, row.ci95_low_auroc, row.ci95_high_auroc, color=COLORS[row.model])
        axes[0].plot(row.mean_auroc, i, "o", color=COLORS[row.model])
        p = matrices[FULL, row.model]
        fpr, tpr, _ = roc_curve(np.tile(y, len(p)), p.ravel())
        axes[1].plot(fpr, tpr, label=row.model, color=COLORS[row.model])
        observed, predicted = calibration_curve(np.tile(y, len(p)), p.ravel(), n_bins=10, strategy="uniform")
        axes[2].plot(predicted, observed, "o-", color=COLORS[row.model], markersize=3)
    axes[0].set(yticks=range(len(s)), yticklabels=s.model, xlabel="Mean repeat AUROC (conditional 95% interval)", xlim=(0, 1))
    axes[0].set_title("Gait + speech + demographics", loc="left")
    for ax in axes[1:]:
        ax.plot([0, 1], [0, 1], "--", color="#a7a7a7", lw=1)
        ax.set(xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Pooled held-out ROC (descriptive)")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[2].set(xlabel="Mean predicted probability", ylabel="Observed fraction", title="Reliability bins (descriptive)")
    for suffix in ("svg", "png"):
        fig.savefig(out / f"overview.{suffix}", dpi=180)
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(stream.getvalue()).decode()


def write_report(out, protocol, summary, delta, matrices, y, frequency, education=None):
    reliability = reliability_rows(matrices, y)
    reliability.to_csv(out / "calibration_bins_by_repeat.csv", index=False)
    calibration = []
    for (feature_set, model), sub in reliability.groupby(["feature_set", "model"], sort=False):
        eces = [float((g.n_subjects * abs(g.mean_probability-g.fraction_positive)).sum() / g.n_subjects.sum())
                for _, g in sub.groupby("repeat")]
        scores = [brier_score_loss(y, p) for p in matrices[feature_set, model]]
        calibration.append({"feature_set": feature_set, "model": model, "mean_brier": np.mean(scores),
                            "mean_10bin_ece": np.mean(eces), "status": "descriptive; no fitted calibrator"})
    calibration = pd.DataFrame(calibration)
    calibration.to_csv(out / "calibration_summary.csv", index=False)
    subgroups = education_groups(out, matrices, y, np.asarray(education), protocol["bootstrap_draws"]) if education is not None else pd.DataFrame()
    with pd.ExcelWriter(out / "analysis_tables.xlsx") as writer:
        for name, data in [("Performance", summary), ("Paired_AUROC", delta), ("Calibration", calibration),
                           ("Feature_recurrence", frequency), ("Education_subgroups", subgroups)]:
            data.to_excel(writer, sheet_name=name, index=False)
    image = make_overview(out, summary, matrices, y)
    display = summary[["feature_set", "model", "mean_auroc", "std_auroc", "ci95_low_auroc", "ci95_high_auroc",
                       "mean_accuracy", "mean_precision", "mean_recall", "mean_f1"]]
    synthetic = "SYNTHETIC DEMONSTRATION: these results contain no study participants." if protocol["synthetic"] else "Research analysis of the supplied feature table."
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ADualNet analysis report</title><style>
body{{font:15px/1.65 'Avenir Next','Trebuchet MS',sans-serif;color:#253d42;background:#f4f3ee;margin:0}}
main{{max-width:1200px;margin:auto;padding:36px}}h1,h2{{font-family:Georgia,serif}}h1{{font-size:42px;margin:0}}
.notice{{background:#e8eee8;border-left:4px solid #367c8d;padding:16px}}section{{background:white;margin:24px 0;padding:24px;border:1px solid #ddd}}
.table{{overflow:auto}}table{{border-collapse:collapse;font-size:12px;width:100%}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #ddd}}th{{background:#edf2f2}}
img{{width:100%}}small{{color:#53676b}}code{{overflow-wrap:anywhere}}a{{color:#256777}}
</style><main><small>ADUALNET / RESEARCH WORKBENCH / {html.escape(protocol['toolkit_version'])}</small>
<h1>Gait, speech &amp; cognitive status</h1><p class="notice">{synthetic}</p>
<p>{protocol['n_subjects']} participants; {protocol['n_splits']} folds × {protocol['n_repeats']} repeats. All feature sets use identical subjects and splits.</p>
<p>Measurement version: <code>{html.escape(protocol['measurement_version'])}</code><br>Outcome: {html.escape(protocol['outcome_definition'])}</p>
<img alt="Model performance, held-out ROC and descriptive calibration" src="data:image/png;base64,{image}">
<section><h2>Performance</h2><p>Metrics are calculated on complete held-out predictions within each repeat, then averaged across repeats. They can differ from historical mean-fold estimates.</p>
<div class="table">{table(display)}</div><p>{CI_NOTE}</p></section>
<section><h2>Paired domain differences</h2><div class="table">{table(delta)}</div><p>Shared participant resamples preserve pairing. Intervals are descriptive conditional comparisons; they do not establish causal domain contributions.</p></section>
<section><h2>Calibration</h2><div class="table">{table(calibration)}</div><p>Reliability curves use pooled repeated held-out probabilities for display. Each repeat has one prediction per participant. Brier scores and ECE are averaged over repeats. Calibration is model-specific; no calibration slope/intercept, probability recalibration or certification is implied.</p></section>
<section><h2>Demographic sensitivity</h2><p>Education-only, age + sex and age + education models use the same cohort and splits as the three-variable demographics model. Adding covariates assesses predictive contribution; this is not causal confounder control.</p>
<p>Selection frequency must be read with the candidate count: demographics-only selects all three candidates by design, so its 100% education recurrence does not independently establish importance.</p>
<div class="table">{table(display[display.feature_set.isin(['demographics-only', 'education-only', 'age + sex', 'age + education'])])}</div>
<h3>Education strata</h3><p>Cutoff fixed at 12 recorded years; estimates are descriptive subsets of full-cohort OOF predictions, without retraining within strata. Cohort imbalance and small class counts limit precision.</p>
<div class="table">{table(subgroups[[c for c in ['education_group','feature_set','model','n_subjects','n_label_0','n_label_1','mean_auroc','ci95_low_auroc','ci95_high_auroc','status'] if c in subgroups]])}</div></section>
<section><h2>Reproducibility and interpretation</h2><p>See protocol.json, analysis_tables.xlsx, feature_mapping.csv, cv_splits_no_id.csv and selected_features_by_fold.csv. Subject-level predictions are pseudonymous local analysis files; keep the full run directory local.</p>
<p>The feature schema follows the study's domains; units and scoring rules must be documented for the supplied measurements. Later raw extractors have not been certified to regenerate the historical manuscript table. Historical performance and data corrections remain separate versions.</p>
<p>This report supports classification of cognitive impairment status within the supplied cohort. Clinical use and external performance require separate evidence.</p></section></main></html>"""
    (out / "report.html").write_text(body, encoding="utf-8")
