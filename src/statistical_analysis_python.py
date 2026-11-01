#!/usr/bin/env python3
"""Python-only statistical analysis and Nature-style figures.

This module replaces the historical R snippets for baseline p-values, group-wise
feature analysis, and automated Pearson/Spearman correlation analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "adualnet_mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "adualnet_xdg_cache"))

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy import stats


RANDOM_STATE = 42

PALETTE = {
    "ink": "#1E252B",
    "muted": "#66717A",
    "grid": "#DDE3E7",
    "blue": "#4C78A8",
    "teal": "#59A14F",
    "orange": "#F28E2B",
    "red": "#E15759",
    "purple": "#8E6C8A",
    "sand": "#EDC948",
    "pale_blue": "#D9E6F2",
    "pale_teal": "#DDEED9",
    "pale_orange": "#F7E2CC",
    "pale_red": "#F3D3D4",
    "white": "#FFFFFF",
}

FEATURE_LABELS = {
    "ACC_DT": "Serial-7 accuracy",
    "Correct_persec_DT": "Correct/sec",
    "Correct_Num_DT": "Correct count",
    "Num_DT": "Serial-7 count",
    "Timecount_DT": "Dual-task time",
    "Timecount_DT_1stNum": "First response time",
    "Num_20s_DT": "Count in 20 s",
    "Correct_Num_20s_DT": "Correct in 20 s",
    "average_step_length": "Step length",
    "average_stride_length": "Stride length",
    "step_length_variability": "Step-length variability",
    "stride_length_variability": "Stride-length variability",
    "average_step_time": "Step time",
    "average_stride_time": "Stride time",
    "step_time_variability": "Step-time variability",
    "stride_time_variability": "Stride-time variability",
    "average_step_assymetry": "Step asymmetry",
    "average_stride_assymetry": "Stride asymmetry",
    "walking_speed": "Walking speed",
    "cadence": "Cadence",
    "stance_times": "Stance time",
    "swing_times": "Swing time",
    "double_support_times": "Double support",
    "age": "Age",
    "edu": "Education",
    "Y_edu": "Education",
    "MMSE": "MMSE",
    "MoCA": "MoCA",
    "gender": "Sex",
}

DEFAULT_BASELINE_CONTINUOUS = ["MMSE", "MoCA", "edu", "age"]
DEFAULT_CATEGORICAL = ["gender"]
DEFAULT_CORRELATION = [
    "ACC_DT",
    "Correct_persec_DT",
    "Correct_Num_DT",
    "Num_DT",
    "Timecount_DT",
    "Timecount_DT_1stNum",
    "Num_20s_DT",
    "Correct_Num_20s_DT",
    "average_step_length",
    "average_stride_length",
    "step_length_variability",
    "stride_length_variability",
    "average_step_time",
    "average_stride_time",
    "step_time_variability",
    "stride_time_variability",
    "average_step_assymetry",
    "average_stride_assymetry",
    "walking_speed",
    "cadence",
    "stance_times",
    "swing_times",
    "double_support_times",
    "age",
    "edu",
    "Y_edu",
    "MMSE",
    "MoCA",
]


@dataclass
class FeatureTestResult:
    variable: str
    label: str
    variable_type: str
    groups: str
    n_total: int
    test: str
    statistic: float
    p_value: float
    effect_size: float
    normality_min_p: float
    levene_p: float
    summary: str


def parse_csv(value: str) -> List[str]:
    return [x.strip() for x in str(value).replace(";", ",").split(",") if x.strip()]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "axes.edgecolor": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_pub(fig: mpl.figure.Figure, out_dir: Path, name: str, dpi: int = 600) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / name
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        color=PALETTE["ink"],
    )


def clean_axis(ax: mpl.axes.Axes, grid_axis: str = "y") -> None:
    ax.tick_params(length=2.5, width=0.65)
    if grid_axis:
        ax.grid(axis=grid_axis, color=PALETTE["grid"], lw=0.55, alpha=0.9)
        ax.set_axisbelow(True)


def wrap_label(text: object, width: int = 18) -> str:
    return "\n".join(wrap(str(text), width=width, break_long_words=False))


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))


def read_table(path: Path, sheet_name: str) -> pd.DataFrame:
    if path.suffix.lower() in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    return pd.read_excel(path, sheet_name=sheet_name)


def parse_group_map(value: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in parse_csv(value):
        if ":" not in item:
            continue
        key, label = item.split(":", 1)
        mapping[key.strip()] = label.strip()
    return mapping


def build_group_series(df: pd.DataFrame, group_col: str, group_map: str, group_order: str) -> Tuple[pd.Series, List[str]]:
    if group_col not in df.columns:
        raise ValueError(f"Group column not found: {group_col}")
    raw = df[group_col]
    mapping = parse_group_map(group_map)

    def convert(value: object) -> object:
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        if text in mapping:
            return mapping[text]
        try:
            numeric_text = str(int(float(text)))
            if numeric_text in mapping:
                return mapping[numeric_text]
        except Exception:
            pass
        return text

    groups = raw.map(convert)
    if group_order:
        order = [x for x in parse_csv(group_order) if x in set(groups.dropna().astype(str))]
    elif mapping:
        order = list(dict.fromkeys(mapping.values()))
        order = [x for x in order if x in set(groups.dropna().astype(str))]
    else:
        order = sorted(groups.dropna().astype(str).unique().tolist())
    groups = pd.Categorical(groups, categories=order, ordered=True)
    return pd.Series(groups, index=df.index, name="Group"), order


def shapiro_p(x: Sequence[float]) -> float:
    arr = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) < 3:
        return float("nan")
    if len(arr) > 5000:
        arr = arr[:5000]
    try:
        return float(stats.shapiro(arr).pvalue)
    except Exception:
        return float("nan")


def is_approximately_normal(x: Sequence[float], alpha: float = 0.05) -> bool:
    arr = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) < 8 or np.nanstd(arr) == 0:
        return False
    p = shapiro_p(arr)
    if math.isfinite(p):
        return p >= alpha
    skew = stats.skew(arr, bias=False, nan_policy="omit")
    kurt = stats.kurtosis(arr, fisher=True, bias=False, nan_policy="omit")
    return abs(float(skew)) < 1.0 and abs(float(kurt)) < 1.0


def outlier_ratio_mad(x: Sequence[float], z_thr: float = 3.5) -> float:
    arr = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return 1.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad == 0:
        return 0.0
    z = np.abs((arr - med) / (1.4826 * mad))
    return float(np.mean(z > z_thr))


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not np.any(valid):
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    tmp = np.empty_like(adj)
    tmp[order] = adj
    out[valid] = tmp
    return out


def p_to_stars(p: float) -> str:
    if not math.isfinite(float(p)):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def mean_sd(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return "NA"
    return f"{x.mean():.2f} ({x.std(ddof=1):.2f})"


def median_iqr(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return "NA"
    q1, q3 = np.percentile(x, [25, 75])
    return f"{np.median(x):.2f} [{q1:.2f}, {q3:.2f}]"


def cohen_d(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(y, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    pooled = math.sqrt(((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1)) / (len(x) + len(y) - 2))
    if pooled == 0:
        return float("nan")
    return float((np.mean(y) - np.mean(x)) / pooled)


def rank_biserial_from_u(u_stat: float, n1: int, n2: int) -> float:
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    return float((2.0 * u_stat) / (n1 * n2) - 1.0)


def eta_squared(groups: Sequence[pd.Series], statistic: float) -> float:
    values = [pd.to_numeric(g, errors="coerce").dropna().to_numpy(dtype=float) for g in groups]
    flat = np.concatenate([v for v in values if len(v)]) if any(len(v) for v in values) else np.array([])
    if flat.size == 0:
        return float("nan")
    overall = np.mean(flat)
    ss_between = sum(len(v) * (np.mean(v) - overall) ** 2 for v in values if len(v))
    ss_total = sum(np.sum((v - overall) ** 2) for v in values if len(v))
    if ss_total == 0:
        return float("nan")
    return float(ss_between / ss_total)


def epsilon_squared_kruskal(h_stat: float, n: int, k: int) -> float:
    if n <= k:
        return float("nan")
    return float(max(0.0, (h_stat - k + 1) / (n - k)))


def continuous_group_test(df: pd.DataFrame, groups: pd.Series, group_order: Sequence[str], variable: str, alpha: float) -> FeatureTestResult:
    frame = pd.DataFrame({"value": pd.to_numeric(df[variable], errors="coerce"), "group": groups}).dropna()
    grouped = [frame.loc[frame["group"].astype(str) == str(g), "value"] for g in group_order]
    grouped = [g for g in grouped if len(g) > 0]
    used_groups = [str(g) for g in group_order if str(g) in set(frame["group"].astype(str))]
    normal_ps = [shapiro_p(g) for g in grouped]
    normal_min = float(np.nanmin(normal_ps)) if any(math.isfinite(p) for p in normal_ps) else float("nan")
    all_normal = all(is_approximately_normal(g, alpha=alpha) for g in grouped) if grouped else False
    levene_p = float("nan")
    if len(grouped) >= 2 and all(len(g) >= 2 for g in grouped):
        try:
            levene_p = float(stats.levene(*grouped, center="median").pvalue)
        except Exception:
            pass
    equal_var = math.isfinite(levene_p) and levene_p >= alpha
    statistic = float("nan")
    p_value = float("nan")
    effect = float("nan")
    test = "not_tested"

    if len(grouped) == 2:
        x, y = grouped
        if len(x) >= 2 and len(y) >= 2 and all_normal:
            res = stats.ttest_ind(x, y, equal_var=equal_var, nan_policy="omit")
            statistic = float(res.statistic)
            p_value = float(res.pvalue)
            effect = cohen_d(x, y)
            test = "Student t-test" if equal_var else "Welch t-test"
        elif len(x) >= 1 and len(y) >= 1:
            res = stats.mannwhitneyu(x, y, alternative="two-sided")
            statistic = float(res.statistic)
            p_value = float(res.pvalue)
            effect = rank_biserial_from_u(float(res.statistic), len(x), len(y))
            test = "Mann-Whitney U"
    elif len(grouped) > 2:
        if all(len(g) >= 2 for g in grouped) and all_normal and equal_var:
            res = stats.f_oneway(*grouped)
            statistic = float(res.statistic)
            p_value = float(res.pvalue)
            effect = eta_squared(grouped, statistic)
            test = "one-way ANOVA"
        elif all(len(g) >= 1 for g in grouped):
            res = stats.kruskal(*grouped)
            statistic = float(res.statistic)
            p_value = float(res.pvalue)
            effect = epsilon_squared_kruskal(float(res.statistic), int(sum(len(g) for g in grouped)), len(grouped))
            test = "Kruskal-Wallis"

    normal_summary = all_normal and (len(grouped) > 0)
    per_group = []
    for group_name, values in zip(used_groups, grouped):
        summary_value = mean_sd(values) if normal_summary else median_iqr(values)
        per_group.append(f"{group_name}: {summary_value}")
    return FeatureTestResult(
        variable=variable,
        label=FEATURE_LABELS.get(variable, variable),
        variable_type="continuous",
        groups="; ".join(used_groups),
        n_total=int(frame["value"].notna().sum()),
        test=test,
        statistic=statistic,
        p_value=p_value,
        effect_size=effect,
        normality_min_p=normal_min,
        levene_p=levene_p,
        summary=" | ".join(per_group),
    )


def categorical_group_test(df: pd.DataFrame, groups: pd.Series, group_order: Sequence[str], variable: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame({"value": df[variable], "group": groups}).dropna()
    table = pd.crosstab(frame["group"], frame["value"]).reindex(group_order, fill_value=0)
    if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
        row = {
            "variable": variable,
            "label": FEATURE_LABELS.get(variable, variable),
            "test": "not_tested",
            "statistic": np.nan,
            "p_value": np.nan,
            "effect_size": np.nan,
            "n_total": int(table.to_numpy().sum()) if not table.empty else 0,
        }
        return pd.DataFrame([row]), table
    chi2, p_value, dof, expected = stats.chi2_contingency(table.to_numpy())
    test = "chi-square"
    statistic = float(chi2)
    effect = math.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))) if min(table.shape) > 1 else float("nan")
    if table.shape == (2, 2) and np.any(expected < 5):
        _, p_value = stats.fisher_exact(table.to_numpy())
        test = "Fisher exact"
        statistic = float("nan")
        effect = float("nan")
    row = {
        "variable": variable,
        "label": FEATURE_LABELS.get(variable, variable),
        "test": test,
        "statistic": statistic,
        "p_value": float(p_value),
        "effect_size": effect,
        "n_total": int(table.to_numpy().sum()),
    }
    return pd.DataFrame([row]), table


def run_continuous_tests(
    df: pd.DataFrame,
    groups: pd.Series,
    group_order: Sequence[str],
    variables: Sequence[str],
    alpha: float,
) -> pd.DataFrame:
    rows = []
    for variable in variables:
        if variable not in df.columns:
            continue
        try:
            rows.append(continuous_group_test(df, groups, group_order, variable, alpha).__dict__)
        except Exception as exc:
            rows.append(
                {
                    "variable": variable,
                    "label": FEATURE_LABELS.get(variable, variable),
                    "variable_type": "continuous",
                    "groups": "; ".join(map(str, group_order)),
                    "n_total": 0,
                    "test": "error",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "effect_size": np.nan,
                    "normality_min_p": np.nan,
                    "levene_p": np.nan,
                    "summary": str(exc),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_fdr_bh"] = bh_fdr(out["p_value"].to_numpy(dtype=float))
        out["sig_stars"] = out["p_value"].map(lambda p: p_to_stars(float(p)) if pd.notna(p) else "")
        out = out.sort_values(["p_fdr_bh", "p_value"], na_position="last").reset_index(drop=True)
    return out


def infer_feature_columns(df: pd.DataFrame, label_col: str, id_col: str, baseline: Sequence[str], categorical: Sequence[str]) -> List[str]:
    excluded = {label_col, id_col, "DX", "Group", "学历"}
    excluded.update(baseline)
    excluded.update(categorical)
    out = []
    for col in df.columns:
        if col in excluded:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() >= 3:
            out.append(col)
    return out


def pairwise_corr_auto(
    df: pd.DataFrame,
    cols: Sequence[str],
    alpha: float = 0.05,
    outlier_thr: float = 0.02,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = df[list(cols)].apply(pd.to_numeric, errors="coerce")
    cols = [c for c in cols if c in data.columns and data[c].notna().sum() >= 3 and data[c].nunique(dropna=True) >= 2]
    n = len(cols)
    r_mat = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    p_mat = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    n_mat = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    method_mat = pd.DataFrame("", index=cols, columns=cols, dtype=object)

    for c in cols:
        r_mat.loc[c, c] = 1.0
        p_mat.loc[c, c] = 0.0
        n_mat.loc[c, c] = data[c].notna().sum()
        method_mat.loc[c, c] = "identity"

    for i, c1 in enumerate(cols):
        for j in range(i + 1, n):
            c2 = cols[j]
            sub = data[[c1, c2]].dropna()
            n_pair = len(sub)
            n_mat.loc[c1, c2] = n_mat.loc[c2, c1] = n_pair
            if n_pair < 3 or sub[c1].nunique() < 2 or sub[c2].nunique() < 2:
                continue
            x, y = sub[c1], sub[c2]
            normal_xy = is_approximately_normal(x, alpha=alpha) and is_approximately_normal(y, alpha=alpha)
            outlier_prop = max(outlier_ratio_mad(x), outlier_ratio_mad(y))
            use_pearson = normal_xy and outlier_prop <= outlier_thr
            if use_pearson:
                r, p = stats.pearsonr(x, y)
                method = "pearson"
            else:
                r, p = stats.spearmanr(x, y)
                method = "spearman"
            r_mat.loc[c1, c2] = r_mat.loc[c2, c1] = float(r)
            p_mat.loc[c1, c2] = p_mat.loc[c2, c1] = float(p)
            method_mat.loc[c1, c2] = method_mat.loc[c2, c1] = method

    fdr_mat = p_mat.copy()
    if n:
        tri = np.tril_indices(n, k=-1)
        p_flat = p_mat.to_numpy()[tri]
        q_flat = bh_fdr(p_flat)
        arr = np.full((n, n), np.nan, dtype=float)
        np.fill_diagonal(arr, 0.0)
        arr[tri] = q_flat
        arr[(tri[1], tri[0])] = q_flat
        fdr_mat = pd.DataFrame(arr, index=cols, columns=cols)
    return r_mat, p_mat, n_mat, method_mat, fdr_mat


def star_matrix(p_mat: pd.DataFrame) -> pd.DataFrame:
    return p_mat.apply(lambda col: col.map(lambda p: p_to_stars(float(p)) if pd.notna(p) else ""))


def write_excel(
    out_dir: Path,
    baseline_cont: pd.DataFrame,
    baseline_cat: pd.DataFrame,
    feature_tests: pd.DataFrame,
    cat_tables: Dict[str, pd.DataFrame],
    r_mat: pd.DataFrame,
    p_mat: pd.DataFrame,
    n_mat: pd.DataFrame,
    method_mat: pd.DataFrame,
    fdr_mat: pd.DataFrame,
) -> None:
    out_path = out_dir / "statistical_analysis_results.xlsx"
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        baseline_cont.to_excel(writer, sheet_name="baseline_continuous", index=False)
        baseline_cat.to_excel(writer, sheet_name="baseline_categorical", index=False)
        feature_tests.to_excel(writer, sheet_name="feature_group_tests", index=False)
        for name, table in cat_tables.items():
            table.to_excel(writer, sheet_name=f"tab_{safe_name(name)[:25]}")
        r_mat.to_excel(writer, sheet_name="corr_r")
        p_mat.to_excel(writer, sheet_name="corr_p")
        fdr_mat.to_excel(writer, sheet_name="corr_p_fdr_bh")
        star_matrix(p_mat).to_excel(writer, sheet_name="corr_sig_stars")
        n_mat.to_excel(writer, sheet_name="corr_N_pairwise")
        method_mat.to_excel(writer, sheet_name="corr_method")


def draw_group_box_panel(
    ax: mpl.axes.Axes,
    df: pd.DataFrame,
    groups: pd.Series,
    group_order: Sequence[str],
    features: Sequence[str],
) -> None:
    if not features:
        ax.text(0.5, 0.5, "No valid feature tests", ha="center", va="center", color=PALETTE["muted"])
        ax.axis("off")
        return
    plot_df = df[list(features)].apply(pd.to_numeric, errors="coerce").copy()
    plot_df["Group"] = groups.astype(str)
    long = plot_df.melt(id_vars="Group", value_vars=list(features), var_name="feature", value_name="value").dropna()
    long["feature_label"] = long["feature"].map(FEATURE_LABELS).fillna(long["feature"])
    labels = [FEATURE_LABELS.get(f, f) for f in features]
    positions = np.arange(len(features))
    width = 0.28 if len(group_order) <= 2 else 0.18
    offsets = np.linspace(-width, width, num=max(len(group_order), 1))
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["teal"], PALETTE["purple"]]
    rng = np.random.default_rng(RANDOM_STATE)
    for gi, group in enumerate(group_order):
        subset = long[long["Group"] == str(group)]
        vals_by_feature = [subset.loc[subset["feature"] == f, "value"].to_numpy(dtype=float) for f in features]
        pos = positions + offsets[gi]
        bp = ax.boxplot(
            vals_by_feature,
            positions=pos,
            widths=width * 0.82,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": PALETTE["ink"], "lw": 0.8},
            whiskerprops={"color": PALETTE["muted"], "lw": 0.7},
            capprops={"color": PALETTE["muted"], "lw": 0.7},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[gi % len(colors)])
            patch.set_alpha(0.32)
            patch.set_edgecolor(colors[gi % len(colors)])
            patch.set_linewidth(0.8)
        for fi, values in enumerate(vals_by_feature):
            if len(values) == 0:
                continue
            jitter = rng.normal(0, width * 0.10, size=len(values))
            ax.scatter(
                np.full(len(values), pos[fi]) + jitter,
                values,
                s=8,
                color=colors[gi % len(colors)],
                alpha=0.55,
                edgecolor="white",
                linewidth=0.2,
                zorder=3,
                label=str(group) if fi == 0 else None,
            )
    ax.set_xticks(positions)
    ax.set_xticklabels([wrap_label(x, 14) for x in labels], rotation=35, ha="right")
    ax.set_ylabel("Feature value")
    ax.legend(loc="upper right", fontsize=6, ncol=min(len(group_order), 3))
    clean_axis(ax, "y")


def draw_single_feature_box(
    ax: mpl.axes.Axes,
    df: pd.DataFrame,
    groups: pd.Series,
    group_order: Sequence[str],
    feature: str,
    p_value: float,
    q_value: float,
) -> None:
    values = pd.to_numeric(df[feature], errors="coerce")
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["teal"], PALETTE["purple"]]
    rng = np.random.default_rng(RANDOM_STATE)
    data = []
    labels = []
    for group in group_order:
        mask = groups.astype(str) == str(group)
        vals = values.loc[mask].dropna().to_numpy(dtype=float)
        data.append(vals)
        labels.append(str(group))
    bp = ax.boxplot(
        data,
        positions=np.arange(len(data)),
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": PALETTE["ink"], "lw": 0.8},
        whiskerprops={"color": PALETTE["muted"], "lw": 0.65},
        capprops={"color": PALETTE["muted"], "lw": 0.65},
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.25)
        patch.set_edgecolor(colors[i % len(colors)])
        patch.set_linewidth(0.75)
    for i, vals in enumerate(data):
        if len(vals) == 0:
            continue
        jitter = rng.normal(0, 0.045, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=5,
            color=colors[i % len(colors)],
            alpha=0.45,
            edgecolor="white",
            linewidth=0.15,
            zorder=3,
        )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_title(
        f"{wrap_label(FEATURE_LABELS.get(feature, feature), 18)}\nP={p_value:.2g}, q={q_value:.2g}",
        loc="left",
        fontsize=6.4,
        fontweight="bold" if q_value < 0.05 else "normal",
        pad=3,
    )
    ax.tick_params(axis="y", labelsize=6)
    clean_axis(ax, "y")


def plot_group_statistics(
    out_dir: Path,
    df: pd.DataFrame,
    groups: pd.Series,
    group_order: Sequence[str],
    baseline_cont: pd.DataFrame,
    baseline_cat: pd.DataFrame,
    feature_tests: pd.DataFrame,
    top_n: int,
) -> None:
    fig = plt.figure(figsize=(7.2, 7.4))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.85, 1.2], width_ratios=[0.95, 1.15], hspace=0.48, wspace=0.38)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    panel_label(ax_a, "a")
    counts = groups.value_counts().reindex(group_order).fillna(0).astype(int)
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["teal"], PALETTE["purple"]]
    ax_a.bar(np.arange(len(counts)), counts.values, color=colors[: len(counts)], alpha=0.85)
    for i, v in enumerate(counts.values):
        ax_a.text(i, v + max(counts.values) * 0.02, str(v), ha="center", va="bottom", fontsize=7)
    ax_a.set_xticks(np.arange(len(counts)))
    ax_a.set_xticklabels(counts.index)
    ax_a.set_ylabel("Participants")
    ax_a.set_title("Analytic groups", loc="left", fontweight="bold", pad=6)
    clean_axis(ax_a, "y")

    panel_label(ax_b, "b")
    base = pd.concat(
        [
            baseline_cont[["label", "p_value"]] if not baseline_cont.empty else pd.DataFrame(columns=["label", "p_value"]),
            baseline_cat[["label", "p_value"]] if not baseline_cat.empty else pd.DataFrame(columns=["label", "p_value"]),
        ],
        ignore_index=True,
    ).dropna(subset=["p_value"])
    if base.empty:
        ax_b.text(0.5, 0.5, "No baseline tests", ha="center", va="center", color=PALETTE["muted"])
        ax_b.axis("off")
    else:
        base["neglog10_p"] = -np.log10(base["p_value"].clip(lower=np.nextafter(0, 1)))
        base = base.sort_values("neglog10_p", ascending=True)
        y = np.arange(len(base))
        ax_b.barh(y, base["neglog10_p"], color=PALETTE["pale_blue"], edgecolor=PALETTE["blue"], linewidth=0.8)
        ax_b.axvline(-math.log10(0.05), color=PALETTE["red"], lw=0.8, ls="--")
        ax_b.set_yticks(y)
        ax_b.set_yticklabels([wrap_label(x, 16) for x in base["label"]])
        ax_b.set_xlabel("-log10(P)")
        ax_b.set_title("Baseline group differences", loc="left", fontweight="bold", pad=6)
        clean_axis(ax_b, "x")

    sig = feature_tests.dropna(subset=["p_value"]).copy()
    sig = sig.sort_values(["p_fdr_bh", "p_value"], na_position="last").head(top_n)
    n_features = len(sig)
    n_cols = 4
    n_rows = max(1, int(math.ceil(max(n_features, 1) / n_cols)))
    subgs = gs[1, :].subgridspec(n_rows, n_cols, hspace=0.78, wspace=0.42)
    mini_axes = []
    for idx in range(n_rows * n_cols):
        ax = fig.add_subplot(subgs[idx // n_cols, idx % n_cols])
        mini_axes.append(ax)
        if idx >= n_features:
            ax.axis("off")
            continue
        row = sig.iloc[idx]
        draw_single_feature_box(
            ax,
            df,
            groups,
            group_order,
            row["variable"],
            float(row["p_value"]),
            float(row["p_fdr_bh"]) if pd.notna(row["p_fdr_bh"]) else float("nan"),
        )
    if mini_axes:
        panel_label(mini_axes[0], "c")
        mini_axes[0].text(
            0.0,
            1.42,
            "Top feature differences by training-free group comparison",
            transform=mini_axes[0].transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=PALETTE["ink"],
        )

    fig.suptitle("Supplementary Figure | Statistical group analysis", x=0.02, ha="left", fontsize=10, fontweight="bold")
    save_pub(fig, out_dir, "figS_statistical_group_analysis")


def select_corr_display(r_mat: pd.DataFrame, p_mat: pd.DataFrame, priority: Sequence[str], max_vars: int) -> List[str]:
    available = [c for c in priority if c in r_mat.columns]
    if len(available) >= max_vars:
        return available[:max_vars]
    remaining = [c for c in r_mat.columns if c not in available]
    if "MoCA" in r_mat.columns:
        score = r_mat.loc[remaining, "MoCA"].abs().sort_values(ascending=False)
    else:
        arr = r_mat.loc[remaining, remaining].abs().replace(1.0, np.nan)
        score = arr.max(axis=1).sort_values(ascending=False)
    available.extend(score.index.tolist())
    return available[:max_vars]


def plot_correlation_figure(
    out_dir: Path,
    r_mat: pd.DataFrame,
    p_mat: pd.DataFrame,
    fdr_mat: pd.DataFrame,
    method_mat: pd.DataFrame,
    max_vars: int,
) -> None:
    if r_mat.empty:
        return
    priority = ["MoCA", "MMSE", "age", "edu", "Y_edu", "ACC_DT", "Correct_persec_DT", "walking_speed", "cadence", "stance_times", "swing_times"]
    cols = select_corr_display(r_mat, p_mat, priority, max_vars)
    r = r_mat.loc[cols, cols]
    p = p_mat.loc[cols, cols]
    q = fdr_mat.loc[cols, cols]
    method = method_mat.loc[cols, cols]

    fig = plt.figure(figsize=(7.2, 6.6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 0.9], width_ratios=[1.15, 0.85], hspace=0.42, wspace=0.36)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    panel_label(ax_a, "a")
    cmap = LinearSegmentedColormap.from_list("nature_corr", [PALETTE["blue"], "#F7F8FA", PALETTE["red"]])
    r_values = r.to_numpy(dtype=float)
    upper_mask = np.triu(np.ones_like(r_values, dtype=bool), k=1)
    masked_r = np.ma.array(r_values, mask=upper_mask)
    cmap.set_bad(color="#FFFFFF")
    im = ax_a.imshow(masked_r, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    labels = [FEATURE_LABELS.get(c, c) for c in cols]
    ax_a.set_xticks(np.arange(len(cols)))
    ax_a.set_yticks(np.arange(len(cols)))
    ax_a.set_xticklabels([wrap_label(x, 12) for x in labels], rotation=45, ha="right")
    ax_a.set_yticklabels([wrap_label(x, 16) for x in labels])
    for i in range(len(cols)):
        for j in range(len(cols)):
            if j > i:
                continue
            if i == j:
                text = "1"
            elif q.iloc[i, j] < 0.05:
                text = f"{r.iloc[i, j]:.2f}{p_to_stars(p.iloc[i, j])}"
            elif p.iloc[i, j] < 0.05:
                text = f"{r.iloc[i, j]:.2f}"
            else:
                text = ""
            if text:
                ax_a.text(j, i, text, ha="center", va="center", fontsize=5.4, color=PALETTE["ink"])
    ax_a.set_title("Pairwise correlation structure", loc="left", fontweight="bold", pad=6)
    cbar = fig.colorbar(im, ax=ax_a, orientation="horizontal", fraction=0.045, pad=0.12)
    cbar.set_label("Correlation coefficient (r)")
    cbar.ax.tick_params(labelsize=6, length=2)
    for spine in ax_a.spines.values():
        spine.set_visible(False)

    panel_label(ax_b, "b")
    pairs = []
    for i in range(len(cols)):
        for j in range(i):
            if math.isfinite(r.iloc[i, j]) and math.isfinite(p.iloc[i, j]):
                pairs.append(
                    {
                        "pair": f"{FEATURE_LABELS.get(cols[i], cols[i])} vs {FEATURE_LABELS.get(cols[j], cols[j])}",
                        "r_abs": abs(float(r.iloc[i, j])),
                        "r": float(r.iloc[i, j]),
                        "p": float(p.iloc[i, j]),
                        "q": float(q.iloc[i, j]) if math.isfinite(q.iloc[i, j]) else np.nan,
                    }
                )
    pair_df = pd.DataFrame(pairs).sort_values(["q", "p", "r_abs"], ascending=[True, True, False]).head(8)
    if pair_df.empty:
        ax_b.text(0.5, 0.5, "No valid pairs", ha="center", va="center", color=PALETTE["muted"])
        ax_b.axis("off")
    else:
        pair_plot = pair_df.iloc[::-1]
        y = np.arange(len(pair_plot))
        colors = [PALETTE["red"] if v > 0 else PALETTE["blue"] for v in pair_plot["r"]]
        ax_b.barh(y, pair_plot["r"], color=colors, alpha=0.85)
        ax_b.axvline(0, color=PALETTE["ink"], lw=0.8)
        ax_b.set_yticks(y)
        ax_b.set_yticklabels([wrap_label(x, 22) for x in pair_plot["pair"]])
        ax_b.set_xlabel("Correlation coefficient")
        ax_b.set_title("Strongest tested associations", loc="left", fontweight="bold", pad=6)
        clean_axis(ax_b, "x")

    panel_label(ax_c, "c")
    method_counts = method.replace("identity", np.nan).stack().value_counts()
    if method_counts.empty:
        ax_c.axis("off")
    else:
        method_counts = method_counts.reindex(["pearson", "spearman"]).dropna()
        ax_c.bar(method_counts.index, method_counts.values, color=[PALETTE["teal"], PALETTE["purple"]][: len(method_counts)], alpha=0.85)
        for i, v in enumerate(method_counts.values):
            ax_c.text(i, v + max(method_counts.values) * 0.02, str(int(v)), ha="center", va="bottom", fontsize=7)
        ax_c.set_ylabel("Pairwise tests")
        ax_c.set_title("Automatic method selection", loc="left", fontweight="bold", pad=6)
        clean_axis(ax_c, "y")

    fig.suptitle("Supplementary Figure | Correlation analysis", x=0.02, ha="left", fontsize=10, fontweight="bold")
    save_pub(fig, out_dir, "figS_correlation_analysis")


def target_columns_for_compact_correlation(r_mat: pd.DataFrame) -> List[str]:
    targets: List[str] = []
    for name in ["MoCA", "MMSE"]:
        if name in r_mat.columns:
            targets.append(name)
            break
    for aliases in [["edu", "Y_edu", "education"], ["age"]]:
        for name in aliases:
            if name in r_mat.columns:
                targets.append(name)
                break
    return targets


def plot_targeted_correlation_panel(
    ax: mpl.axes.Axes,
    fig: mpl.figure.Figure,
    r_mat: pd.DataFrame,
    p_mat: pd.DataFrame,
    fdr_mat: pd.DataFrame,
    features: Sequence[str],
) -> None:
    targets = target_columns_for_compact_correlation(r_mat)
    rows = [feature for feature in features if feature in r_mat.index and feature not in targets]
    if not rows or not targets:
        ax.text(0.5, 0.5, "No valid targeted correlations", ha="center", va="center", color=PALETTE["muted"])
        ax.axis("off")
        return
    r = r_mat.loc[rows, targets]
    p = p_mat.loc[rows, targets]
    q = fdr_mat.loc[rows, targets]
    cmap = LinearSegmentedColormap.from_list("target_corr", [PALETTE["blue"], "#F7F8FA", PALETTE["red"]])
    im = ax.imshow(r.to_numpy(dtype=float), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(targets)))
    ax.set_xticklabels([wrap_label(FEATURE_LABELS.get(x, x), 12) for x in targets])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([wrap_label(FEATURE_LABELS.get(x, x), 22) for x in rows])
    for i in range(len(rows)):
        for j in range(len(targets)):
            rval = r.iloc[i, j]
            pval = p.iloc[i, j]
            qval = q.iloc[i, j]
            if not math.isfinite(float(rval)):
                continue
            stars = p_to_stars(float(pval)) if math.isfinite(float(pval)) else ""
            weight = "bold" if math.isfinite(float(qval)) and qval < 0.05 else "normal"
            ax.text(
                j,
                i,
                f"{rval:.2f}{stars}",
                ha="center",
                va="center",
                fontsize=6.3,
                fontweight=weight,
                color=PALETTE["ink"],
            )
    ax.set_title("Targeted correlations with cognition and demographics", loc="left", fontweight="bold", pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.035, pad=0.015)
    cbar.set_label("r", rotation=0, labelpad=8)
    cbar.ax.tick_params(labelsize=6, length=2)
    ax.text(
        0.0,
        -0.15,
        "Cells show automatically selected Pearson or Spearman r; stars indicate nominal two-sided P values.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        color=PALETTE["muted"],
    )


def plot_combined_correlation_group_figure(
    out_dir: Path,
    df: pd.DataFrame,
    groups: pd.Series,
    group_order: Sequence[str],
    feature_tests: pd.DataFrame,
    r_mat: pd.DataFrame,
    p_mat: pd.DataFrame,
    fdr_mat: pd.DataFrame,
    top_n: int,
) -> None:
    ranked = feature_tests.dropna(subset=["p_value"]).copy()
    ranked = ranked.sort_values(["p_fdr_bh", "p_value"], na_position="last").head(top_n)
    features = ranked["variable"].tolist()
    if not features:
        return

    n_cols = 4
    n_rows = max(1, int(math.ceil(len(features) / n_cols)))
    fig = plt.figure(figsize=(7.2, 8.2))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.2, 2.35], hspace=0.42)

    ax_corr = fig.add_subplot(gs[0, 0])
    panel_label(ax_corr, "a")
    plot_targeted_correlation_panel(ax_corr, fig, r_mat, p_mat, fdr_mat, features)

    subgs = gs[1, 0].subgridspec(n_rows, n_cols, hspace=0.78, wspace=0.42)
    mini_axes = []
    for idx in range(n_rows * n_cols):
        ax = fig.add_subplot(subgs[idx // n_cols, idx % n_cols])
        mini_axes.append(ax)
        if idx >= len(ranked):
            ax.axis("off")
            continue
        row = ranked.iloc[idx]
        draw_single_feature_box(
            ax,
            df,
            groups,
            group_order,
            row["variable"],
            float(row["p_value"]),
            float(row["p_fdr_bh"]) if pd.notna(row["p_fdr_bh"]) else float("nan"),
        )
    if mini_axes:
        panel_label(mini_axes[0], "b")
        mini_axes[0].text(
            0.0,
            1.42,
            "Group differences for the same feature set",
            transform=mini_axes[0].transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=PALETTE["ink"],
        )

    fig.suptitle(
        "Supplementary Figure | Statistical associations and group differences",
        x=0.02,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    save_pub(fig, out_dir, "figS_statistical_correlation_group_combined")


def write_manifest(out_dir: Path, args: argparse.Namespace, group_order: Sequence[str], feature_tests: pd.DataFrame, r_mat: pd.DataFrame) -> None:
    manifest = {
        "core_conclusion": "Statistical analyses quantify group differences and pairwise associations among gait, speech, demographic and cognitive variables.",
        "backend": "python/matplotlib",
        "input": str(Path(args.input).expanduser().resolve()),
        "sheet_name": args.sheet_name,
        "group_col": args.group_col,
        "group_order": list(group_order),
        "alpha": args.alpha,
        "n_feature_tests": int(len(feature_tests)),
        "n_correlation_variables": int(len(r_mat.columns)),
        "outputs": [
            "statistical_analysis_results.xlsx",
            "baseline_continuous.csv",
            "baseline_categorical.csv",
            "feature_group_tests.csv",
            "correlation_r.csv",
            "correlation_p.csv",
            "correlation_p_fdr_bh.csv",
            "figS_statistical_group_analysis.svg/pdf/png/tiff",
            "figS_correlation_analysis.svg/pdf/png/tiff",
            "figS_statistical_correlation_group_combined.svg/pdf/png/tiff",
        ],
        "review_risks": [
            "Group comparisons are unadjusted descriptive analyses unless covariates are explicitly modelled elsewhere.",
            "Correlation analyses use pairwise deletion and should not be interpreted as causal effects.",
            "FDR-adjusted p-values are reported alongside nominal p-values for multiplicity control.",
        ],
    }
    (out_dir / "statistical_figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Python-only statistical tests and Nature-style figures.")
    parser.add_argument("--input", required=True, help="Input xlsx/csv containing features and labels.")
    parser.add_argument("--sheet-name", default="info")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--id-col", default="ID")
    parser.add_argument("--group-col", default="label")
    parser.add_argument("--group-map", default="0:CU,1:CI")
    parser.add_argument("--group-order", default="")
    parser.add_argument("--baseline-continuous", default=",".join(DEFAULT_BASELINE_CONTINUOUS))
    parser.add_argument("--categorical-vars", default=",".join(DEFAULT_CATEGORICAL))
    parser.add_argument("--feature-cols", default="", help="Comma-separated feature columns. Empty means infer numeric non-baseline features.")
    parser.add_argument("--correlation-cols", default=",".join(DEFAULT_CORRELATION))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--outlier-thr", type=float, default=0.02)
    parser.add_argument("--top-boxplot-features", type=int, default=12)
    parser.add_argument("--max-corr-vars", type=int, default=12)
    args = parser.parse_args()

    setup_style()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_table(Path(args.input).expanduser().resolve(), args.sheet_name)
    groups, group_order = build_group_series(df, args.group_col, args.group_map, args.group_order)
    valid_mask = groups.notna()
    df = df.loc[valid_mask].copy()
    groups = groups.loc[valid_mask]

    baseline_cont_vars = [c for c in parse_csv(args.baseline_continuous) if c in df.columns]
    categorical_vars = [c for c in parse_csv(args.categorical_vars) if c in df.columns]
    feature_cols = parse_csv(args.feature_cols)
    if not feature_cols:
        feature_cols = infer_feature_columns(df, args.label_col, args.id_col, baseline_cont_vars, categorical_vars)
    else:
        feature_cols = [c for c in feature_cols if c in df.columns]
    corr_cols = [c for c in parse_csv(args.correlation_cols) if c in df.columns]
    if not corr_cols:
        corr_cols = feature_cols + baseline_cont_vars

    baseline_cont = run_continuous_tests(df, groups, group_order, baseline_cont_vars, args.alpha)
    feature_tests = run_continuous_tests(df, groups, group_order, feature_cols, args.alpha)

    cat_rows = []
    cat_tables: Dict[str, pd.DataFrame] = {}
    for variable in categorical_vars:
        row, table = categorical_group_test(df, groups, group_order, variable)
        cat_rows.append(row)
        cat_tables[variable] = table
    baseline_cat = pd.concat(cat_rows, ignore_index=True) if cat_rows else pd.DataFrame()
    if not baseline_cat.empty:
        baseline_cat["p_fdr_bh"] = bh_fdr(baseline_cat["p_value"].to_numpy(dtype=float))
        baseline_cat["sig_stars"] = baseline_cat["p_value"].map(lambda p: p_to_stars(float(p)) if pd.notna(p) else "")

    r_mat, p_mat, n_mat, method_mat, fdr_mat = pairwise_corr_auto(df, corr_cols, alpha=args.alpha, outlier_thr=args.outlier_thr)

    baseline_cont.to_csv(out_dir / "baseline_continuous.csv", index=False)
    baseline_cat.to_csv(out_dir / "baseline_categorical.csv", index=False)
    feature_tests.to_csv(out_dir / "feature_group_tests.csv", index=False)
    r_mat.to_csv(out_dir / "correlation_r.csv")
    p_mat.to_csv(out_dir / "correlation_p.csv")
    fdr_mat.to_csv(out_dir / "correlation_p_fdr_bh.csv")
    n_mat.to_csv(out_dir / "correlation_N_pairwise.csv")
    method_mat.to_csv(out_dir / "correlation_method.csv")

    write_excel(out_dir, baseline_cont, baseline_cat, feature_tests, cat_tables, r_mat, p_mat, n_mat, method_mat, fdr_mat)
    plot_group_statistics(out_dir, df, groups, group_order, baseline_cont, baseline_cat, feature_tests, args.top_boxplot_features)
    plot_correlation_figure(out_dir, r_mat, p_mat, fdr_mat, method_mat, args.max_corr_vars)
    plot_combined_correlation_group_figure(
        out_dir,
        df,
        groups,
        group_order,
        feature_tests,
        r_mat,
        p_mat,
        fdr_mat,
        args.top_boxplot_features,
    )
    write_manifest(out_dir, args, group_order, feature_tests, r_mat)
    print(json.dumps({"out_dir": str(out_dir), "n_rows": int(len(df)), "groups": {str(k): int(v) for k, v in groups.value_counts().items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
