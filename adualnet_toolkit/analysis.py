"""Versioned feature-table analyses reusing the paper's model/ranking code."""

import hashlib
import csv
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from . import __version__
from .uncertainty import METRICS, bootstrap_weights, summarize_predictions

PUBLIC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC_ROOT / "src"))
from domain_ablation_analysis import DOMAIN_SPECS, FEATURE_SETS, PAIRED_COMPARISONS
from paper_experiment_reproduction import (
    make_publication_classification_models, predict_probability,
    safe_fit, select_top_k_features_train_only,
)

ALIASES = {
    "sex": ["sex", "gender"], "edu": ["edu", "education"],
    "average_step_assymetry": ["average_step_assymetry", "average_step_asymmetry"],
    "average_stride_assymetry": ["average_stride_assymetry", "average_stride_asymmetry"],
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_table(path, sheet="0"):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
        if len(set(header)) != len(header):
            raise ValueError("Duplicate CSV headers must be resolved before analysis.")
        return pd.read_csv(path, dtype={"ID": str})
    if path.suffix.lower() == ".xlsx":
        key = int(sheet) if str(sheet).isdigit() else sheet
        first = pd.read_excel(path, sheet_name=key, header=None, nrows=1)
        header = first.iloc[0].dropna().astype(str).tolist() if len(first) else []
        if len(set(header)) != len(header):
            raise ValueError("Duplicate Excel headers must be resolved before analysis.")
        return pd.read_excel(path, sheet_name=key, dtype={"ID": str})
    raise ValueError("Choose a CSV or XLSX feature table.")


def prepare_table(frame, n_splits=5, id_col="ID", label_col="label"):
    errors, warnings, mapping = [], [], []
    if frame.columns.duplicated().any():
        errors.append("Duplicate column names are not allowed.")
    canonical = pd.DataFrame(index=frame.index)
    for domain, features in DOMAIN_SPECS.items():
        for name in features:
            aliases = ALIASES.get(name, [name])
            matches = [c for c in frame.columns if str(c).lower() in [a.lower() for a in aliases]]
            status = "matched" if len(matches) == 1 else "missing" if not matches else "ambiguous"
            mapping.append({"domain": domain, "feature": name, "source_column": str(matches[0]) if len(matches) == 1 else "", "status": status})
            if status != "matched":
                errors.append(f"{name}: {status}; expected one of {aliases}.")
                continue
            raw = frame[matches[0]]
            numeric = pd.to_numeric(raw, errors="coerce")
            bad = raw.notna() & numeric.isna()
            if bad.any():
                errors.append(f"{name}: {int(bad.sum())} nonnumeric values. Encode sex numerically with a recorded codebook.")
            if np.isinf(numeric).any():
                errors.append(f"{name}: infinite values must be reviewed.")
            if not numeric.notna().any():
                errors.append(f"{name}: all values are missing.")
            canonical[name] = numeric
    if id_col not in frame:
        errors.append(f"Participant column {id_col!r} is required.")
    else:
        ids = frame[id_col]
        if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
            errors.append("Participant IDs must be nonempty.")
        if ids.astype(str).str.strip().duplicated().any():
            errors.append("Use one row per participant; repeated recordings must be resolved before this analysis.")
    if label_col not in frame:
        errors.append(f"Outcome column {label_col!r} is required.")
        y = pd.Series(dtype=float)
    else:
        y = pd.to_numeric(frame[label_col], errors="coerce")
        if y.isna().any() or not y.isin([0, 1]).all() or set(y.dropna()) != {0, 1}:
            errors.append("Every outcome must be exactly 0 or 1, and both classes must be present.")
        elif y.value_counts().min() < n_splits:
            errors.append(f"Each outcome class needs at least {n_splits} participants.")
    missingness = [{"feature": c, "missing_count": int(canonical[c].isna().sum()), "constant": bool(canonical[c].nunique() <= 1)} for c in canonical]
    if any(row["missing_count"] for row in missingness):
        warnings.append("Missing values will be imputed from each training fold; no participants are dropped by feature set.")
    if any(row["constant"] for row in missingness):
        warnings.append("Constant features were found; their univariate ranking score is zero.")
    unused = [str(c) for c in frame.columns if c not in [m["source_column"] for m in mapping] and c not in (id_col, label_col)]
    warnings.append("Feature names do not certify units or historical extraction equivalence; record measurement definitions separately.")
    report = {"passed": not errors, "errors": errors, "warnings": warnings, "n_subjects": len(frame),
              "class_counts": {str(k): int(v) for k, v in y.value_counts().items()},
              "mapping": mapping, "missingness": missingness, "unused_columns": unused}
    return canonical, y, report


def feature_sets(include_demographics=True):
    sets = {name: [f for domain in domains for f in DOMAIN_SPECS[domain]] for name, domains in FEATURE_SETS.items()}
    if include_demographics:
        sets.update({"education-only": ["edu"], "age + sex": ["age", "sex"], "age + education": ["age", "edu"]})
    return sets


def make_demo(n=80, seed=1729):
    """Entirely synthetic measurements; no fitted patient-derived generator."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"ID": [f"SYNTHETIC_{i + 1:04d}" for i in range(n)]})
    latent = rng.normal(size=n)
    frame["age"] = np.round(rng.normal(69, 6, n), 1)
    frame["sex"] = rng.integers(0, 2, n)
    frame["edu"] = np.clip(np.round(rng.normal(11, 3, n)), 0, 22)
    frame["walking_speed"] = np.clip(1.1 - 0.12 * latent + rng.normal(0, 0.16, n), 0.3, 2)
    frame["cadence"] = np.clip(rng.normal(102, 12, n) - latent * 4, 65, 145)
    frame["average_step_length"] = frame.walking_speed * 60 / frame.cadence
    frame["average_stride_length"] = 2 * frame.average_step_length
    frame["average_step_time"] = 60 / frame.cadence
    frame["average_stride_time"] = 2 * frame.average_step_time
    for f in DOMAIN_SPECS["gait"][6:]:
        frame[f] = np.clip(rng.lognormal(-2, 0.35, n) + latent * 0.015, 0.01, 1)
    frame["stance_times"] = 0.60 * frame.average_stride_time
    frame["swing_times"] = 0.40 * frame.average_stride_time
    frame["double_support_times"] = 0.20 * frame.average_stride_time
    frame["Timecount_DT"] = np.round(rng.uniform(15, 40, n), 2)
    frame["Num_DT"] = np.maximum(0, np.round(14 - latent * 3 + rng.normal(0, 3, n))).astype(int)
    frame["Correct_Num_DT"] = np.maximum(0, frame.Num_DT - rng.integers(0, 4, n))
    frame["Num_20s_DT"] = np.floor(frame.Num_DT * np.minimum(20 / frame.Timecount_DT, 1)).astype(int)
    frame["Correct_Num_20s_DT"] = np.minimum(frame.Correct_Num_DT, frame.Num_20s_DT)
    frame["Correct_persec_DT"] = frame.Correct_Num_DT / frame.Timecount_DT
    frame["ACC_DT"] = np.divide(frame.Correct_Num_DT, frame.Num_DT, out=np.zeros(n), where=frame.Num_DT > 0)
    frame["Timecount_DT_1stNum"] = np.clip(2.5 + latent + rng.normal(0, 0.7, n), 0.1, 8)
    logit = latent * 1.1 + (frame.age - 69) * 0.04 - (frame.edu - 11) * 0.11
    frame["label"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return frame


def run_analysis(input_path, out_dir, *, sheet="0", n_splits=5, n_repeats=10, bootstrap=2000,
                 models=("AdaBoost", "RF", "SVM (RBF)", "DT"), select_k=14,
                 measurement_version="", outcome_definition="", synthetic=False, log=print):
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory already contains files. Choose a new run directory.")
    out.mkdir(parents=True, exist_ok=True)
    log(f"[INFO] ADualNet {__version__}")
    try:
        if not (2 <= n_splits <= 10 and 2 <= n_repeats <= 20 and 20 <= bootstrap <= 10000 and select_k >= 1):
            raise ValueError("Use 2-10 folds, 2-20 repeats, 20-10000 bootstrap draws, and top-k >= 1.")
        if not measurement_version.strip() or not outcome_definition.strip():
            raise ValueError("Record a measurement version and the meaning of outcome 0/1 before analysis.")
        protos = make_publication_classification_models()
        if not models or any(m not in protos for m in models) or len(set(models)) != len(models):
            raise ValueError("Choose unique models from AdaBoost, RF, SVM (RBF), DT.")
        frame = load_table(input_path, sheet)
        X, y_series, preflight = prepare_table(frame, n_splits)
        write_json(out / "preflight.json", preflight)
        (out / "preflight_check.txt").write_text("PASS\n" if preflight["passed"] else "FAIL\n" + "\n".join(preflight["errors"]), encoding="utf-8")
        if preflight["errors"]:
            raise ValueError("Input check failed: " + "; ".join(preflight["errors"]))
        y = y_series.to_numpy(dtype=int)
        n = len(y)
        anon_ids = [f"subject_{i + 1:05d}" for i in range(n)]
        sets = feature_sets()
        versions = {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "matplotlib", "openpyxl"]}
        protocol = {
            "toolkit_version": __version__, "analysis": "full-cohort within-task domain ablation",
            "input_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
            "measurement_version": measurement_version, "outcome_definition": outcome_definition,
            "synthetic": synthetic, "n_subjects": n, "class_counts": preflight["class_counts"],
            "n_splits": n_splits, "n_repeats": n_repeats, "seed": 42, "top_k_cap": select_k,
            "models": list(models), "feature_sets": sets, "threshold": 0.5,
            "ranking": "abs(training-only univariate AUROC - 0.5)",
            "preprocessing": "training-fold median imputation; training-fold scaling where applicable",
            "ci_method": "outcome-stratified participant cluster percentile bootstrap of cached held-out predictions",
            "bootstrap_draws": bootstrap, "bootstrap_seed": 20261031, "confidence_level": 0.95,
            "estimand": "mean of complete out-of-fold metrics across repeats (not mean of 50 fold metrics)",
            "ci_conditions": "all repeated predictions stay with their participant; same draws for all comparisons; no refitting",
            "ci_limitations": "conditional on fitted cross-validation models and observed class counts; shared training-set dependence and model-selection uncertainty are not fully captured; not population-performance or external-validation guarantees",
            "sd_definition": "sample SD across repeated complete OOF metric estimates; repeats reuse participants",
            "probability_calibration": "descriptive reliability bins and Brier; no post-hoc calibrator fitted",
            "demographics": "predictor addition/removal, not causal confounder adjustment",
            "versions": versions, "python": platform.python_version(),
            "source_sha256": {name: hashlib.sha256((PUBLIC_ROOT / name).read_bytes()).hexdigest()
                              for name in ["adualnet_toolkit/analysis.py", "adualnet_toolkit/uncertainty.py",
                                           "adualnet_toolkit/report.py", "src/paper_experiment_reproduction.py",
                                           "src/domain_ablation_analysis.py"]},
            "model_parameters": {m: {k: repr(v) for k, v in protos[m].get_params(deep=True).items()} for m in models},
        }
        write_json(out / "used_config.json", protocol)
        write_json(out / "protocol.json", protocol)
        pd.DataFrame(preflight["mapping"]).to_csv(out / "feature_mapping.csv", index=False)
        pd.DataFrame(preflight["missingness"]).to_csv(out / "missingness.csv", index=False)
        splits = list(RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42).split(X, y))
        split_rows = [{"repeat": i // n_splits + 1, "fold": i % n_splits + 1, "anon_subject_id": anon_ids[j], "role": role}
                      for i, (tr, te) in enumerate(splits) for role, indices in [("train", tr), ("test", te)] for j in indices]
        pd.DataFrame(split_rows).to_csv(out / "cv_splits_no_id.csv", index=False)
        matrices, folds, selections, preds = {}, [], [], []
        total = len(sets) * len(splits) * len(models)
        done = 0
        for set_name, columns in sets.items():
            arr = X[columns].to_numpy(float)
            for model_name in models:
                matrices[set_name, model_name] = np.full((n_repeats, n), np.nan)
            for split_idx, (tr, te) in enumerate(splits):
                repeat, fold = split_idx // n_splits, split_idx % n_splits + 1
                selected, _ = select_top_k_features_train_only(arr[tr], y[tr], columns, min(select_k, len(columns)))
                names = [columns[k] for k in selected]
                selections.extend({"feature_set": set_name, "repeat": repeat + 1, "fold": fold, "feature": c} for c in names)
                for model_name in models:
                    est = safe_fit(clone(protos[model_name]), arr[tr][:, selected], y[tr])
                    p = predict_probability(est, arr[te][:, selected])
                    matrices[set_name, model_name][repeat, te] = p
                    folds.append({"feature_set": set_name, "model": model_name, "repeat": repeat + 1, "fold": fold,
                                  "auroc": float(roc_auc_score(y[te], p)), "n_test": len(te)})
                    preds.extend({"feature_set": set_name, "model": model_name, "repeat": repeat + 1, "fold": fold,
                                  "anon_subject_id": anon_ids[j], "y_true": int(y[j]), "y_prob": float(prob), "y_pred": int(prob >= .5)}
                                 for j, prob in zip(te, p))
                    done += 1
                    log(f"[RUN] {done}/{total}: {set_name} / {model_name} / repeat {repeat + 1}, fold {fold}")
        pd.DataFrame(folds).to_csv(out / "fold_metrics.csv", index=False)
        pd.DataFrame(preds).to_csv(out / "predictions_no_id.csv", index=False)
        selection_df = pd.DataFrame(selections)
        selection_df.to_csv(out / "selected_features_by_fold.csv", index=False)
        frequency = selection_df.groupby(["feature_set", "feature"]).size().reset_index(name="selected_fold_count")
        frequency["selected_fold_fraction"] = frequency.selected_fold_count / len(splits)
        frequency.to_csv(out / "selected_feature_frequency.csv", index=False)
        weights = bootstrap_weights(y, bootstrap, 20261031)
        summary_rows, repeat_rows, boot_results, confusion_rows = [], [], {}, []
        for (set_name, model_name), p in matrices.items():
            log(f"[RUN] participant bootstrap: {set_name} / {model_name}")
            summary, per_repeat, boots = summarize_predictions(y, p, weights)
            summary_rows.append({"feature_set": set_name, "model": model_name, "n_subjects": n, **summary})
            boot_results[set_name, model_name] = boots
            for repeat, values in enumerate(per_repeat, start=1):
                repeat_rows.append({"feature_set": set_name, "model": model_name, "repeat": repeat, **dict(zip(METRICS, values))})
            for repeat, pp in enumerate(p, start=1):
                tn, fp, fn, tp = confusion_matrix(y, pp >= .5, labels=[0, 1]).ravel()
                confusion_rows.append({"feature_set": set_name, "model": model_name, "repeat": repeat,
                                       "tn": tn, "fp": fp, "fn": fn, "tp": tp, "n_subjects": n})
        summary_df = pd.DataFrame(summary_rows)
        repeat_df = pd.DataFrame(repeat_rows)
        deltas = []
        for comparison, a, b in PAIRED_COMPARISONS:
            for model_name in models:
                d = boot_results[a, model_name][:, 0] - boot_results[b, model_name][:, 0]
                means = summary_df.set_index(["feature_set", "model"])["mean_auroc"]
                deltas.append({"comparison": comparison, "model": model_name, "delta_auroc": means[a, model_name] - means[b, model_name],
                               "ci95_low": np.quantile(d, .025), "ci95_high": np.quantile(d, .975)})
        delta_df = pd.DataFrame(deltas)
        summary_df.to_csv(out / "performance.csv", index=False)
        repeat_df.to_csv(out / "repeat_metrics.csv", index=False)
        delta_df.to_csv(out / "paired_delta_auroc.csv", index=False)
        pd.DataFrame(confusion_rows).to_csv(out / "confusion_by_repeat.csv", index=False)
        from .report import write_report
        write_report(out, protocol, summary_df, delta_df, matrices, y, frequency, X.edu.to_numpy())
        write_json(out / "COMPLETE.json", {"status": "complete", "toolkit_version": __version__, "synthetic": synthetic, "n_subjects": n})
        log("[DONE] ALL ANALYSES COMPLETED")
        return out / "report.html"
    except Exception as exc:
        (out / "FAILED_STEP_README.txt").write_text(f"ADualNet {__version__}\n{type(exc).__name__}: {exc}\nReview preflight.json and used_config.json if present. Choose a new output folder for retry.\n", encoding="utf-8")
        raise
