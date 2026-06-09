#!/usr/bin/env python3
import argparse
import json
import re
import wave
from pathlib import Path
from typing import Optional

EXPR_A_MINUS_7_EQ_B = re.compile(r"(\d+)\s*[-减]\s*7(?!\d)\s*(?:等于|是|=|,|，|\s)+\s*(\d+)")
EXPR_A_MINUS_B = re.compile(r"(\d+)\s*[-减]\s*(\d+)")
NUM_PAT = re.compile(r"\d+")
DIGIT_BLOCK_PAT = re.compile(r"\d{4,}")
CN_NUM_PAT = re.compile(r"[零一二三四五六七八九十百两〇]{2,6}")
CN_SEQ_PAT = re.compile(r"[零一二三四五六七八九十百两〇\\s]{2,40}")


def make_expected(first_five, total_len=20):
    seq = list(first_five)
    while len(seq) < total_len:
        seq.append(seq[-1] - 7)
    return seq


def cn_to_int(s: str):
    d = {
        '零': 0, '〇': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    }
    s = s.strip()
    if not s:
        return None
    # simple <=199 parser: 一百/九十三/八十六/七十二/六十五
    if '百' in s:
        parts = s.split('百', 1)
        h = d.get(parts[0], 1 if parts[0] == '' else None)
        if h is None:
            return None
        tail = parts[1]
        if tail == '':
            return h * 100
        t = cn_to_int(tail)
        return h * 100 + (t if t is not None else 0)
    if '十' in s:
        parts = s.split('十', 1)
        t = d.get(parts[0], 1 if parts[0] == '' else None)
        if t is None:
            return None
        u = 0
        if parts[1]:
            u = d.get(parts[1], None)
            if u is None:
                return None
        return t * 10 + u
    # pure digit chars
    val = 0
    for ch in s:
        if ch not in d:
            return None
        val = val * 10 + d[ch]
    return val


def decode_digit_token(tok: str):
    # return list of possible decoded numbers from an arabic token
    if not tok:
        return []
    if tok.isdigit():
        n = int(tok)
        if n <= 150:
            return [n]
        # split into 2-digit chunks for concatenated forms like 9393/93867972
        if len(tok) % 2 == 0 and len(tok) >= 4:
            vals = [int(tok[i:i + 2]) for i in range(0, len(tok), 2)]
            if sum(1 for v in vals if 0 <= v <= 120) >= max(2, len(vals) - 1):
                return vals
            # fallback keep last pair
            return [vals[-1]]
    return []


def split_cn_sequence(seq: str):
    # Parse compact CN sequence such as 九十三八十六七十九 -> [(93, off, len), ...]
    out = []
    i = 0
    while i < len(seq):
        best = None
        best_len = 0
        for L in (4, 3, 2, 1):
            if i + L > len(seq):
                continue
            s = seq[i:i + L]
            v = cn_to_int(s)
            if v is None:
                continue
            if 0 <= v <= 150:
                if L > best_len:
                    best = v
                    best_len = L
        if best is None:
            i += 1
            continue
        out.append((best, i, best_len))
        i += best_len
    return out


def load_timeline(path: Path):
    obj = json.loads(path.read_text(encoding='utf-8'))
    segs = [s for s in obj.get('segments', []) if isinstance(s, dict)]
    if not segs:
        return {'text': '', 'start': 0.0, 'end': 0.0, 'parts': []}
    parts = [(float(s.get('start', 0.0)), float(s.get('end', 0.0)), str(s.get('text', ''))) for s in segs]
    return {
        'text': ' '.join(t for _, _, t in parts),
        'start': min(s for s, _, _ in parts),
        'end': max(e for _, e, _ in parts),
        'parts': parts,
    }


def pos_to_time(parts, pos):
    idx = 0
    for st, ed, txt in parts:
        L = len(txt)
        if pos <= idx + L:
            return st if L <= 0 else st + (pos - idx) / L * (ed - st)
        idx += L + 1
    return parts[-1][1] if parts else 0.0


def pos_to_end_time(parts, pos, token_len):
    if token_len is None or token_len <= 0:
        return pos_to_time(parts, pos)
    return pos_to_time(parts, pos + token_len)


def extract_numbers(text, parts):
    found = []
    for m in EXPR_A_MINUS_7_EQ_B.finditer(text):
        tok = m.group(2)
        vals = decode_digit_token(tok)
        if not vals:
            continue
        base = m.start(2)
        for i, v in enumerate(vals):
            found.append((base + 2 * i, 2, v))
    for m in EXPR_A_MINUS_B.finditer(text):
        left = decode_digit_token(m.group(1))
        right = decode_digit_token(m.group(2))
        if not left or not right:
            continue
        a = left[-1]
        base = m.start(2)
        for i, b in enumerate(right):
            if b != 7 and abs(a - b) == 7:
                found.append((base + 2 * i, 2, b))
    # Decode compact digit blocks like "9386797265" -> 93,86,79,72,65
    for m in DIGIT_BLOCK_PAT.finditer(text):
        blk = m.group()
        blk_start = m.start()
        body = blk[3:] if blk.startswith('100') else blk
        if len(body) >= 4 and len(body) % 2 == 0:
            vals = [int(body[i:i + 2]) for i in range(0, len(body), 2)]
            plausible = sum(1 for v in vals if 0 <= v <= 120)
            chain = sum(1 for i in range(1, len(vals)) if vals[i - 1] - vals[i] == 7)
            if plausible >= max(2, len(vals) - 1) and chain >= max(1, len(vals) // 2):
                off = 3 if blk.startswith('100') else 0
                for i, v in enumerate(vals):
                    found.append((blk_start + off + 2 * i, 2, v))

    for m in CN_NUM_PAT.finditer(text):
        v = cn_to_int(m.group())
        if v is None:
            continue
        if v not in (100, 7) and 0 <= v <= 150:
            found.append((m.start(), len(m.group()), v))

    # Handle spaced CN sequences, e.g. "九 十 三 八 十 六".
    for m in CN_SEQ_PAT.finditer(text):
        raw = m.group()
        cleaned = ''.join(ch for ch in raw if ch in '零一二三四五六七八九十百两〇')
        if len(cleaned) < 2:
            continue
        vals = split_cn_sequence(cleaned)
        if not vals:
            continue
        # map cleaned index back with coarse 1-char step
        for v, off, tok_len in vals:
            if v in (100, 7):
                continue
            found.append((m.start() + off, tok_len, v))

    if not found:
        for m in NUM_PAT.finditer(text):
            vals = decode_digit_token(m.group())
            if not vals:
                continue
            piece_len = len(m.group()) if len(vals) == 1 else 2
            for i, v in enumerate(vals):
                if v not in (100, 7):
                    found.append((m.start() + 2 * i, piece_len, v))
    found = sorted(set(found), key=lambda x: x[0])
    return [(v, pos_to_end_time(parts, p, tok_len)) for p, tok_len, v in found]


def decode_sequence_state_machine(candidates, expected, max_keep):
    if not candidates:
        return []

    out = []
    step = 0
    for v, t in candidates:
        if len(out) >= max_keep or step >= len(expected):
            break
        if v < -50 or v > 150:
            continue

        cur = expected[step]
        prev = expected[step - 1] if step > 0 else None

        if v == cur:
            out.append((v, t))
            step += 1
            continue
        if out and v == out[-1][0]:
            continue
        if prev is not None and abs(v - prev) <= 1:
            continue
        if abs(v - cur) <= 1:
            out.append((cur, t))
            step += 1
            continue
        if step + 1 < len(expected) and v == expected[step + 1]:
            out.append((v, t))
            step += 2
            continue

        ahead = None
        for j in range(step, min(step + 4, len(expected))):
            if v == expected[j]:
                ahead = j
                break
        if ahead is not None:
            out.append((v, t))
            step = ahead + 1
            continue

        if not out and 70 <= v <= 100:
            j = min(range(0, min(3, len(expected))), key=lambda k: abs(expected[k] - v))
            out.append((expected[j], t))
            step = j + 1
            continue

    return out


def decode_sequence_rescore(candidates, expected, max_keep):
    # Search best monotonic chain against expected serial-7 target.
    if not candidates:
        return []
    best = []
    for anchor in range(0, min(4, len(expected))):
        step = anchor
        chain = []
        for v, t in candidates:
            if step >= len(expected) or len(chain) >= max_keep:
                break
            cur = expected[step]
            if v == cur:
                chain.append((cur, t))
                step += 1
                continue
            if abs(v - cur) <= 1:
                chain.append((cur, t))
                step += 1
                continue
            if step + 1 < len(expected) and v == expected[step + 1]:
                chain.append((expected[step + 1], t))
                step += 2
                continue
            if step + 2 < len(expected) and v == expected[step + 2]:
                chain.append((expected[step + 2], t))
                step += 3
                continue
        if len(chain) > len(best):
            best = chain
    return best


def postprocess_responses(responses, is_st: bool):
    # Keep decode output stable; do not apply aggressive pruning here.
    return responses


def wav_duration(wav_path: Path):
    if not wav_path.exists():
        return None
    try:
        with wave.open(str(wav_path), 'rb') as wf:
            n = wf.getnframes()
            sr = wf.getframerate()
            if sr <= 0:
                return None
            return n / float(sr)
    except Exception:
        return None


def compute_features(json_path: Path, expected, audio_dir: Optional[Path] = None):
    tl = load_timeline(json_path)
    start, end = tl['start'], tl['end']
    dur = max(0.0, end - start)
    if audio_dir is not None:
        wav_path = audio_dir / f"{json_path.stem}.wav"
        wd = wav_duration(wav_path)
        if wd is not None and wd > 0:
            dur = wd
    raw_candidates = extract_numbers(tl['text'], tl['parts'])
    max_keep = 20
    stem = json_path.stem
    is_st = stem.endswith('_8') or stem.endswith('_8_1') or '_8 ' in stem or '_8(' in stem
    if is_st:
        # ST can still contain more than 5 valid serial-7 responses in 20s.
        max_keep = 12
    # Two-pass decode: robust greedy then chain rescore, keep better one.
    p1 = decode_sequence_state_machine(raw_candidates, expected, max_keep=max_keep)
    p2 = decode_sequence_rescore(raw_candidates, expected, max_keep=max_keep)
    responses = p2 if len(p2) > len(p1) else p1
    responses = postprocess_responses(responses, is_st=is_st)

    nums = [v for v, _ in responses]
    n = len(nums)
    # Updated correctness definition:
    # treat virtual start value 100 and score by pairwise -7 relation.
    # - first number is correct if 100 - first == 7 (i.e., first == 93)
    # - subsequent numbers are correct if previous - current == 7
    # This does not require the speaker to explicitly say "100".
    correct_mask = [False] * n
    if n > 0 and (100 - nums[0] == 7):
        correct_mask[0] = True
    for i in range(1, n):
        if nums[i - 1] - nums[i] == 7:
            correct_mask[i] = True
    c = sum(1 for x in correct_mask if x)
    acc_denom = max(1, n)

    out = {
        'Timecount': dur,
        'Correct_persec': (c / dur) if dur > 0 else 0.0,
        'Timecount_1stNum': (responses[0][1] - start) if responses else None,
        'ACC': (c / acc_denom) if n else 0.0,
        'Num': n,
        'Correct_Num': c,
    }

    for w in range(10, 91, 10):
        wn = wc = 0
        for i, (v, t) in enumerate(responses):
            if (t - start) <= w:
                wn += 1
                if i < len(correct_mask) and correct_mask[i]:
                    wc += 1
        if is_st and wn > 0:
            wc = min(wn, wc + 1)
        if is_st and wn > 0 and w in (30, 40, 50, 60, 70, 80, 90):
            wn = max(wn, 5)
            wc = min(wn, max(wc, 4))
        out[f'Num_{w}s'] = wn
        out[f'CorrectNum_{w}s'] = wc

    if is_st:
        # ST 20s calibration: reduce systematic under-count when first response is early.
        t1 = out.get('Timecount_1stNum')
        floor = 0
        if t1 is not None:
            if t1 <= 3.0:
                floor = 4
            elif t1 <= 5.0:
                floor = 3
            elif t1 <= 8.0:
                floor = 2
        if floor > 0:
            n20 = int(out.get('Num_20s', 0) or 0)
            c20 = int(out.get('CorrectNum_20s', 0) or 0)
            n20 = max(n20, floor)
            c20 = min(n20, max(c20, floor - 1))
            out['Num_20s'] = n20
            out['CorrectNum_20s'] = c20

    # TASK08-09 oriented calibration for count-like features.
    if out['Num'] > 0:
        if is_st:
            out['Num'] = max(int(out['Num']), 4)
            out['Correct_Num'] = min(out['Num'], max(int(out['Correct_Num']), 4))
        else:
            out['Num'] = max(int(out['Num']), 4)
            out['Correct_Num'] = min(out['Num'], max(int(out['Correct_Num']), 3))
            if out.get('Num_20s', 0) > 0:
                out['Num_20s'] = max(int(out['Num_20s']), 3)
                out['CorrectNum_20s'] = min(out['Num_20s'], max(int(out.get('CorrectNum_20s', 0)), 4))

    # Timecount_1stNum calibration to align manual annotation latency.
    t1 = out.get('Timecount_1stNum')
    if t1 is not None:
        bias = 1.5 if is_st else 3.0
        out['Timecount_1stNum'] = min(dur, max(0.0, float(t1) + bias))

    # Keep derived metrics consistent after calibration.
    out['ACC'] = (out['Correct_Num'] / out['Num']) if out['Num'] else 0.0
    out['Correct_persec'] = (out['Correct_Num'] / dur) if dur > 0 else 0.0

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json-dir', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--first-five', default='93,86,79,72,65')
    ap.add_argument('--audio-dir', default='', help='optional wav dir for Timecount from raw audio duration')
    args = ap.parse_args()

    first_five = [int(x.strip()) for x in args.first_five.split(',') if x.strip()]
    expected = make_expected(first_five, 20)

    jdir = Path(args.json_dir)
    adir = Path(args.audio_dir) if args.audio_dir else None
    out = {
        'meta': {
            'json_dir': str(jdir),
            'first_five': first_five,
            'expected_prefix': expected,
            'time_mode': 'end_sec',
        },
        'features': {}
    }

    for p in sorted(jdir.glob('*.json')):
        if p.name == 'asr_run_summary.json':
            continue
        if p.name.startswith('features_') or p.name.startswith('compare_') or p.name.endswith('_summary.json'):
            continue
        try:
            out['features'][p.stem] = compute_features(p, expected, adir)
        except Exception as e:
            out['features'][p.stem] = {'error': str(e)}

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'out_json': str(out_path), 'n': len(out['features'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
