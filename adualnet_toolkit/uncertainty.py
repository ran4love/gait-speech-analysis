"""Conditional participant bootstrap of cached repeated held-out predictions.

One row of each matrix is one CV repeat, one column is one participant.
All repeats and model comparisons share participant resamples. Models are not
refitted, so these intervals do not quantify training/selection uncertainty.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

METRICS = ("auroc", "accuracy", "precision", "recall", "f1", "brier")


def repeat_metrics(y, probabilities, weights=None):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(y) or not np.isfinite(p).all():
        raise ValueError("Each CV repeat must have one finite held-out probability per participant.")
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != y.shape or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Invalid participant bootstrap weights.")
    positive = (y == 1)[None, :]
    pred = p >= 0.5
    tp = ((pred & positive) * w).sum(axis=1)
    fp = ((pred & ~positive) * w).sum(axis=1)
    fn = ((~pred & positive) * w).sum(axis=1)
    tn = ((~pred & ~positive) * w).sum(axis=1)

    def divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)

    return np.column_stack([
        [roc_auc_score(y, row, sample_weight=w) for row in p],
        (tp + tn) / w.sum(), divide(tp, tp + fp), divide(tp, tp + fn),
        divide(2 * tp, 2 * tp + fp + fn), ((p - y) ** 2 * w).sum(axis=1) / w.sum(),
    ])


def bootstrap_weights(y, count, seed):
    """Stratify by recorded outcome; preserve its observed counts in each draw."""
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    weights = np.zeros((count, len(y)), dtype=int)
    for label in (0, 1):
        positions = np.flatnonzero(y == label)
        if not len(positions):
            raise ValueError("Both outcome classes are required for AUROC.")
        for b in range(count):
            sampled = rng.choice(positions, size=len(positions), replace=True)
            weights[b] += np.bincount(sampled, minlength=len(y))
    return weights


def summarize_predictions(y, probabilities, weights):
    per_repeat = repeat_metrics(y, probabilities)
    # Vectorize bootstrap metrics; ties are integrated exactly in weighted AUC.
    y = np.asarray(y, dtype=int)
    w = np.asarray(weights, dtype=float)
    totals = w.sum(axis=1)
    boot = np.zeros((len(w), len(METRICS)))
    for p in np.asarray(probabilities):
        order = np.argsort(p, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
        pos = np.add.reduceat(w[:, order] * y[order], starts, axis=1)
        neg = np.add.reduceat(w[:, order] * (1 - y[order]), starts, axis=1)
        auc = (pos * (np.cumsum(neg, axis=1) - .5 * neg)).sum(axis=1) / (pos.sum(axis=1) * neg.sum(axis=1))
        pred = p >= .5
        tp = w @ (pred & (y == 1)).astype(float)
        fp = w @ (pred & (y == 0)).astype(float)
        fn = w @ (~pred & (y == 1)).astype(float)
        tn = w @ (~pred & (y == 0)).astype(float)
        def div(a, b):
            return np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        boot += np.column_stack([auc, (tp + tn) / totals, div(tp, tp + fp), div(tp, tp + fn),
                                 div(2 * tp, 2 * tp + fp + fn), (w @ ((p - y) ** 2)) / totals])
    boot /= len(probabilities)
    summary = {}
    for index, metric in enumerate(METRICS):
        summary.update({
            f"mean_{metric}": float(per_repeat[:, index].mean()),
            f"std_{metric}": float(per_repeat[:, index].std(ddof=1)) if len(per_repeat) > 1 else 0.0,
            f"ci95_low_{metric}": float(np.quantile(boot[:, index], 0.025)),
            f"ci95_high_{metric}": float(np.quantile(boot[:, index], 0.975)),
        })
    return summary, per_repeat, boot
