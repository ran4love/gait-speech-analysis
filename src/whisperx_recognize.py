#!/usr/bin/env python3
import argparse
import json
import re
import time
from pathlib import Path

import torch
import whisperx


def preset_asr_options(preset: str):
    # Keep only options supported by the server whisperx/faster-whisper build.
    base = {
        'word_timestamps': True,
        'max_new_tokens': 448,
        'clip_timestamps': '0',
        'hallucination_silence_threshold': None,
    }
    if preset == 'st':
        # TASK08: shorter response chain, prefer precision.
        base.update({
            'beam_size': 6,
            'no_speech_threshold': 0.55,
            'compression_ratio_threshold': 2.2,
            'condition_on_previous_text': False,
        })
        return base
    if preset == 'dt':
        # TASK09: longer chain, keep context continuity.
        base.update({
            'beam_size': 5,
            'no_speech_threshold': 0.60,
            'compression_ratio_threshold': 2.4,
            'condition_on_previous_text': True,
        })
        return base
    # default
    base.update({
        'beam_size': 5,
        'no_speech_threshold': 0.60,
        'compression_ratio_threshold': 2.4,
        'condition_on_previous_text': False,
    })
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--model-name', default='large-v2')
    ap.add_argument('--language', default='zh')
    ap.add_argument('--compute-type', default='float16')
    ap.add_argument('--max-files', type=int, default=0)
    ap.add_argument('--preset', default='default', choices=['default', 'st', 'dt'])
    ap.add_argument('--name-regex', default='', help='optional regex on wav filename')
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    asr_options = preset_asr_options(args.preset)

    model = whisperx.load_model(
        args.model_name,
        language=args.language,
        device=device,
        compute_type=args.compute_type,
        asr_options=asr_options,
    )
    model_a, metadata = whisperx.load_align_model(language_code=args.language, device=device)

    files = sorted(in_dir.glob('*.wav'))
    if args.name_regex:
        pat = re.compile(args.name_regex)
        files = [p for p in files if pat.search(p.name)]
    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]

    summary = {
        'input_dir': str(in_dir),
        'output_dir': str(out_dir),
        'model_name': args.model_name,
        'language': args.language,
        'device': device,
        'preset': args.preset,
        'name_regex': args.name_regex,
        'total_files': len(files),
        'success': 0,
        'failed': 0,
        'items': [],
    }

    for wav in files:
        t0 = time.time()
        out_json = out_dir / f'{wav.stem}.json'
        item = {'file': wav.name, 'json': out_json.name, 'status': 'ok', 'error': '', 'sec': 0.0}
        try:
            audio = whisperx.load_audio(str(wav))
            result = model.transcribe(audio)
            try:
                aligned = whisperx.align(result['segments'], model_a, metadata, audio, device)
            except Exception:
                # Keep raw ASR result if alignment fails
                aligned = {'segments': result.get('segments', []), 'word_segments': result.get('word_segments', [])}
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(aligned, f, ensure_ascii=False, indent=2)
            summary['success'] += 1
        except Exception as e:
            item['status'] = 'failed'
            item['error'] = str(e)
            summary['failed'] += 1

        item['sec'] = round(time.time() - t0, 3)
        summary['items'].append(item)

    with open(out_dir / 'asr_run_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({'total': summary['total_files'], 'success': summary['success'], 'failed': summary['failed']}))


if __name__ == '__main__':
    main()
