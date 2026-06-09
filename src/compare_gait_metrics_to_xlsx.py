#!/usr/bin/env python3
"""Compare gait metrics across pred_2025, pred_2026, and check_previous.xlsx.

This script uses only Python standard library and produces an .xlsx report.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import zipfile
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET


XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XLSX_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
DC_NS = "http://purl.org/dc/elements/1.1/"
CP_NS = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DCTERMS_NS = "http://purl.org/dc/terms/"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
EXT_PROPS_NS = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
VT_NS = "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"

METRICS = ["stance_times", "swing_times", "double_support_times"]
EPS = 1e-12


def normalize_case_id(value: object) -> str:
    s = str(value or "").strip()
    if s.lower().endswith(".json"):
        s = s[:-5]
    if s.lower().endswith(".mp4"):
        s = s[:-4]
    return s


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def col_name_to_index(name: str) -> int:
    n = 0
    for ch in name:
        if "A" <= ch <= "Z":
            n = n * 26 + (ord(ch) - ord("A") + 1)
    return n


def index_to_col_name(idx: int) -> str:
    out = []
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        out.append(chr(ord("A") + rem))
    return "".join(reversed(out))


def parse_xlsx_first_sheet(path: Path) -> Tuple[List[str], List[Dict[str, object]]]:
    ns = {"a": XLSX_NS, "r": XLSX_REL_NS}
    with zipfile.ZipFile(path) as zf:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            sst_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in sst_root.findall("a:si", ns):
                texts = [t.text or "" for t in si.findall(".//a:t", ns)]
                shared_strings.append("".join(texts))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        sheets = workbook.findall("a:sheets/a:sheet", ns)
        if not sheets:
            raise ValueError("No sheets found in workbook.")
        first_sheet = sheets[0]
        rel_id = first_sheet.attrib.get(f"{{{XLSX_REL_NS}}}id")
        if not rel_id:
            raise ValueError("Cannot resolve first sheet relationship id.")

        wb_rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_target = None
        for rel in wb_rels.findall(f"{{{PKG_REL_NS}}}Relationship"):
            if rel.attrib.get("Id") == rel_id:
                rel_target = rel.attrib.get("Target")
                break
        if not rel_target:
            raise ValueError("Cannot resolve first sheet target.")
        sheet_path = "xl/" + rel_target
        sheet_root = ET.fromstring(zf.read(sheet_path))

    rows_raw: List[Dict[int, object]] = []
    for row in sheet_root.findall(".//a:sheetData/a:row", ns):
        row_map: Dict[int, object] = {}
        for c in row.findall("a:c", ns):
            ref = c.attrib.get("r", "")
            col = "".join(ch for ch in ref if ch.isalpha())
            col_idx = col_name_to_index(col)
            cell_type = c.attrib.get("t")
            val: object = ""
            v = c.find("a:v", ns)
            if cell_type == "s" and v is not None and v.text is not None:
                idx = int(v.text)
                val = shared_strings[idx] if 0 <= idx < len(shared_strings) else ""
            elif cell_type == "inlineStr":
                is_tag = c.find("a:is", ns)
                if is_tag is not None:
                    val = "".join((t.text or "") for t in is_tag.findall(".//a:t", ns))
                else:
                    val = ""
            elif v is not None and v.text is not None:
                val = v.text
            row_map[col_idx] = val
        rows_raw.append(row_map)

    if not rows_raw:
        return [], []

    max_col = max((max(r.keys()) if r else 0) for r in rows_raw)
    header_row = rows_raw[0]
    headers = [str(header_row.get(i, "")).strip() for i in range(1, max_col + 1)]

    records: List[Dict[str, object]] = []
    for rr in rows_raw[1:]:
        rec = {}
        for i, h in enumerate(headers, start=1):
            if h:
                rec[h] = rr.get(i, "")
        records.append(rec)
    return headers, records


def load_ds_json(path: Path) -> Dict[str, Dict[str, Optional[float]]]:
    raw = json.loads(path.read_text())
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for case_id, obj in raw.items():
        cid = normalize_case_id(case_id)
        if not cid:
            continue
        metrics = {}
        for m in METRICS:
            metrics[m] = to_float(obj.get(m) if isinstance(obj, dict) else None)
        out[cid] = metrics
    return out


def load_previous_metrics(path: Path) -> Dict[str, Dict[str, Optional[float]]]:
    headers, rows = parse_xlsx_first_sheet(path)
    if "ID" not in headers:
        raise ValueError("ID column not found in check_previous.xlsx first sheet.")

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for row in rows:
        cid = normalize_case_id(row.get("ID", ""))
        if not cid:
            continue
        metrics = {m: to_float(row.get(m)) for m in METRICS}
        out[cid] = metrics
    return out


def relative_similarity(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    denom = abs(a) + abs(b)
    if denom <= EPS:
        return 1.0
    sim = 1.0 - abs(a - b) / denom
    return max(0.0, min(1.0, sim))


def cosine_similarity(a: Sequence[Optional[float]], b: Sequence[Optional[float]]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None
    dot = sum(x * y for x, y in pairs)
    na = math.sqrt(sum(x * x for x, _ in pairs))
    nb = math.sqrt(sum(y * y for _, y in pairs))
    if na <= EPS or nb <= EPS:
        return None
    return max(0.0, min(1.0, dot / (na * nb)))


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return statistics.mean(vals)


def safe_median(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return statistics.median(vals)


def compare_sets(
    left_name: str,
    right_name: str,
    left_data: Dict[str, Dict[str, Optional[float]]],
    right_data: Dict[str, Dict[str, Optional[float]]],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    left_ids = set(left_data.keys())
    right_ids = set(right_data.keys())
    common = sorted(left_ids & right_ids)

    details: List[Dict[str, object]] = []
    overall_scores: List[float] = []

    for cid in common:
        lvals = [left_data[cid].get(m) for m in METRICS]
        rvals = [right_data[cid].get(m) for m in METRICS]

        sim_by_metric = [relative_similarity(a, b) for a, b in zip(lvals, rvals)]
        overall = safe_mean(sim_by_metric)
        cos_sim = cosine_similarity(lvals, rvals)
        if overall is not None:
            overall_scores.append(overall * 100.0)

        row: Dict[str, object] = {
            "comparison": f"{left_name}_vs_{right_name}",
            "case_id": cid,
            "overall_similarity_pct": round((overall or 0.0) * 100.0, 6) if overall is not None else None,
            "cosine_similarity_pct": round((cos_sim or 0.0) * 100.0, 6) if cos_sim is not None else None,
        }

        for m, a, b, sm in zip(METRICS, lvals, rvals, sim_by_metric):
            row[f"{left_name}_{m}"] = a
            row[f"{right_name}_{m}"] = b
            row[f"abs_diff_{m}"] = abs(a - b) if a is not None and b is not None else None
            row[f"sim_{m}_pct"] = round((sm or 0.0) * 100.0, 6) if sm is not None else None

        details.append(row)

    if overall_scores:
        mean_score = statistics.mean(overall_scores)
        median_score = statistics.median(overall_scores)
        min_score = min(overall_scores)
        max_score = max(overall_scores)
    else:
        mean_score = median_score = min_score = max_score = None

    summary = {
        "comparison": f"{left_name}_vs_{right_name}",
        "left_cases": len(left_ids),
        "right_cases": len(right_ids),
        "common_cases": len(common),
        "left_only": len(left_ids - right_ids),
        "right_only": len(right_ids - left_ids),
        "mean_overall_similarity_pct": round(mean_score, 6) if mean_score is not None else None,
        "median_overall_similarity_pct": round(median_score, 6) if median_score is not None else None,
        "min_overall_similarity_pct": round(min_score, 6) if min_score is not None else None,
        "max_overall_similarity_pct": round(max_score, 6) if max_score is not None else None,
    }
    return summary, details


def make_sheet_xml(rows: Sequence[Sequence[object]]) -> str:
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        f'<worksheet xmlns="{XLSX_NS}">',
        "<sheetData>",
    ]

    for r_idx, row in enumerate(rows, start=1):
        xml_parts.append(f'<row r="{r_idx}">')
        for c_idx, val in enumerate(row, start=1):
            if val is None or val == "":
                continue
            ref = f"{index_to_col_name(c_idx)}{r_idx}"
            if isinstance(val, bool):
                val = 1 if val else 0
            if isinstance(val, (int, float)):
                if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                    sval = escape(str(val))
                    xml_parts.append(f'<c r="{ref}" t="inlineStr"><is><t>{sval}</t></is></c>')
                else:
                    xml_parts.append(f'<c r="{ref}"><v>{val}</v></c>')
            else:
                sval = escape(str(val))
                xml_parts.append(f'<c r="{ref}" t="inlineStr"><is><t>{sval}</t></is></c>')
        xml_parts.append("</row>")

    xml_parts.append("</sheetData></worksheet>")
    return "".join(xml_parts)


def write_xlsx(path: Path, sheets: Sequence[Tuple[str, Sequence[Sequence[object]]]]) -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
        '<Override PartName="/docProps/core.xml" '
        'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
        '<Override PartName="/docProps/app.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
    ]
    for i in range(1, len(sheets) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    pkg_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{PKG_REL_NS}">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
        'Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
        'Target="docProps/app.xml"/>'
        "</Relationships>"
    )

    wb_rels_parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        f'<Relationships xmlns="{PKG_REL_NS}">',
    ]
    for i in range(1, len(sheets) + 1):
        wb_rels_parts.append(
            f'<Relationship Id="rId{i}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{i}.xml"/>'
        )
    wb_rels_parts.append(
        f'<Relationship Id="rId{len(sheets)+1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    wb_rels_parts.append("</Relationships>")

    workbook_parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        f'<workbook xmlns="{XLSX_NS}" xmlns:r="{XLSX_REL_NS}">',
        "<sheets>",
    ]
    for i, (sheet_name, _) in enumerate(sheets, start=1):
        safe_name = escape(sheet_name)
        workbook_parts.append(f'<sheet name="{safe_name}" sheetId="{i}" r:id="rId{i}"/>')
    workbook_parts.append("</sheets></workbook>")

    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<styleSheet xmlns="{XLSX_NS}">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/><family val="2"/></font></fonts>'
        '<fills count="2"><fill><patternFill patternType="none"/></fill>'
        '<fill><patternFill patternType="gray125"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        "</styleSheet>"
    )

    core = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<cp:coreProperties xmlns:cp="{CP_NS}" xmlns:dc="{DC_NS}" '
        f'xmlns:dcterms="{DCTERMS_NS}" xmlns:xsi="{XSI_NS}">'
        "<dc:creator>Codex</dc:creator>"
        "<cp:lastModifiedBy>Codex</cp:lastModifiedBy>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
        "</cp:coreProperties>"
    )

    heading_pairs = (
        '<vt:vector size="2" baseType="variant">'
        '<vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant>'
        f'<vt:variant><vt:i4>{len(sheets)}</vt:i4></vt:variant>'
        "</vt:vector>"
    )
    titles = [f"<vt:lpstr>{escape(name)}</vt:lpstr>" for name, _ in sheets]
    titles_xml = f'<vt:vector size="{len(sheets)}" baseType="lpstr">' + "".join(titles) + "</vt:vector>"
    app = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Properties xmlns="{EXT_PROPS_NS}" xmlns:vt="{VT_NS}">'
        "<Application>Codex</Application>"
        f"<HeadingPairs>{heading_pairs}</HeadingPairs>"
        f"<TitlesOfParts>{titles_xml}</TitlesOfParts>"
        "</Properties>"
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "".join(content_types))
        zf.writestr("_rels/.rels", pkg_rels)
        zf.writestr("xl/workbook.xml", "".join(workbook_parts))
        zf.writestr("xl/_rels/workbook.xml.rels", "".join(wb_rels_parts))
        zf.writestr("xl/styles.xml", styles)
        zf.writestr("docProps/core.xml", core)
        zf.writestr("docProps/app.xml", app)
        for i, (_, rows) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", make_sheet_xml(rows))


def rows_from_dicts(dict_rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> List[List[object]]:
    out: List[List[object]] = [list(columns)]
    for d in dict_rows:
        out.append([d.get(c) for c in columns])
    return out


def build_output(
    pred25_path: Path,
    pred26_path: Path,
    previous_xlsx_path: Path,
    output_path: Path,
) -> None:
    pred25 = load_ds_json(pred25_path)
    pred26 = load_ds_json(pred26_path)
    previous = load_previous_metrics(previous_xlsx_path)

    summary_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []

    for left_name, right_name, left_data, right_data in [
        ("pred_2025", "pred_2026", pred25, pred26),
        ("pred_2025", "previous", pred25, previous),
        ("pred_2026", "previous", pred26, previous),
    ]:
        summary, details = compare_sets(left_name, right_name, left_data, right_data)
        summary_rows.append(summary)
        detail_rows.extend(details)

    summary_cols = [
        "comparison",
        "left_cases",
        "right_cases",
        "common_cases",
        "left_only",
        "right_only",
        "mean_overall_similarity_pct",
        "median_overall_similarity_pct",
        "min_overall_similarity_pct",
        "max_overall_similarity_pct",
    ]

    detail_cols = [
        "comparison",
        "case_id",
        "overall_similarity_pct",
        "cosine_similarity_pct",
        "pred_2025_stance_times",
        "pred_2026_stance_times",
        "previous_stance_times",
        "abs_diff_stance_times",
        "sim_stance_times_pct",
        "pred_2025_swing_times",
        "pred_2026_swing_times",
        "previous_swing_times",
        "abs_diff_swing_times",
        "sim_swing_times_pct",
        "pred_2025_double_support_times",
        "pred_2026_double_support_times",
        "previous_double_support_times",
        "abs_diff_double_support_times",
        "sim_double_support_times_pct",
    ]

    detail_rows = sorted(
        detail_rows,
        key=lambda d: (
            str(d.get("comparison", "")),
            -(float(d["overall_similarity_pct"]) if d.get("overall_similarity_pct") is not None else -1),
            str(d.get("case_id", "")),
        ),
    )

    summary_sheet = rows_from_dicts(summary_rows, summary_cols)
    detail_sheet = rows_from_dicts(detail_rows, detail_cols)

    write_xlsx(output_path, [("Summary", summary_sheet), ("Details", detail_sheet)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare gait metrics and export xlsx report.")
    parser.add_argument(
        "--pred-2025",
        default="gait_features_pred_2025_ds.json",
        help="Path to pred_2025 ds features JSON.",
    )
    parser.add_argument(
        "--pred-2026",
        default="gait_features_pred_2026_ds.json",
        help="Path to pred_2026 ds features JSON.",
    )
    parser.add_argument(
        "--previous-xlsx",
        default="check_previous.xlsx",
        help="Path to previous metrics xlsx.",
    )
    parser.add_argument(
        "--output-xlsx",
        default="gait_metrics_similarity_comparison.xlsx",
        help="Output comparison xlsx path.",
    )
    args = parser.parse_args()

    pred_2025_path = Path(args.pred_2025)
    pred_2026_path = Path(args.pred_2026)
    previous_path = Path(args.previous_xlsx)
    output_path = Path(args.output_xlsx)

    for p in [pred_2025_path, pred_2026_path, previous_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    build_output(pred_2025_path, pred_2026_path, previous_path, output_path)
    print(f"Comparison xlsx written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
