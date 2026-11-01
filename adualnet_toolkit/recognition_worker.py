#!/usr/bin/env python3
"""Model adapters run in the configured recognition Python, including Python 3.8.

No expected number sequence or outcome is supplied to either neural model.
"""

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import sys
from pathlib import Path


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    return value


def runtime(config, packages):
    info = {"requested": config, "python": sys.version, "packages": {}, "explicit_weight_files": {}}
    for name in packages:
        try:
            info["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][name] = "not installed under this distribution name"
    for key in ("pose2d_weights", "det_weights"):
        path = Path(config.get(key, ""))
        if config.get(key) and path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(chunk)
            info["explicit_weight_files"][key] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    return info


def pose(video, config, out):
    import cv2
    from mmpose.apis import MMPoseInferencer
    if config.get("det_weights") and not config.get("det_model"):
        raise ValueError("Explicit detector weights require an explicit det_model configuration.")
    kwargs = {k: config[k] for k in ["pose2d", "pose2d_weights", "det_model", "det_weights", "device"] if config.get(k)}
    inferencer = MMPoseInferencer(**kwargs)
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    frames = []
    for index, result in enumerate(inferencer(str(video), show=False, return_vis=False, bbox_thr=float(config["bbox_thr"]))):
        pred = plain(result.get("predictions", []))
        instances = pred[0] if pred and isinstance(pred[0], list) else pred
        frames.append({"frame_index": index, "time_s": index/fps, "instances": instances})
        if index % 90 == 0:
            print("[RUN] MMPose frame {}".format(index), flush=True)
    if not frames:
        raise ValueError("MMPose produced no frames")
    try:
        import mmengine
        engine = inferencer.inferencer
        resolved = {**runtime(config, ["mmpose", "mmcv", "mmengine", "mmdet", "torch"]),
                    "pose_config": engine.cfg.pretty_text, "pose_config_file": engine.cfg.filename,
                    "mmpose_version": __import__("mmpose").__version__,
                    "mmengine_version": mmengine.__version__, "requested": config}
        if getattr(engine, "detector", None) is not None:
            resolved["detector_config"] = engine.detector.cfg.pretty_text
            resolved["detector_config_file"] = engine.detector.cfg.filename
    except Exception as exc:
        resolved = {**runtime(config, ["mmpose", "torch"]), "resolved_config_error": str(exc)}
    out.write_text(json.dumps({"frames": frames, "fps": fps, "provenance": resolved}, ensure_ascii=False), encoding="utf-8")


def speech(audio, config, out):
    if config["backend"] == "faster-whisper":
        from faster_whisper import WhisperModel
        model = WhisperModel(config["model"], device=config["device"], compute_type=config["compute_type"])
        stream, info = model.transcribe(str(audio), language=config.get("language") or None,
                                       beam_size=5, word_timestamps=True, condition_on_previous_text=False)
        segments = []
        for segment in stream:
            words = [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in (segment.words or [])]
            segments.append({"start": segment.start, "end": segment.end, "text": segment.text, "words": words})
            print("[RUN] Whisper segment {:.2f}-{:.2f}s".format(segment.start, segment.end), flush=True)
        result = {"segments": segments, "language": info.language}
    elif config["backend"] == "whisperx":
        import whisperx
        model = whisperx.load_model(config["model"], device=config["device"],
                                    language=config.get("language") or None, compute_type=config["compute_type"])
        data = whisperx.load_audio(str(audio))
        result = model.transcribe(data)
        # Alignment may download a separate model; failed alignment is retained as
        # segment-level timing for explicit manual review rather than hidden.
        try:
            aligner, metadata = whisperx.load_align_model(language_code=result.get("language", config["language"]), device=config["device"])
            result = whisperx.align(result["segments"], aligner, metadata, data, config["device"])
        except Exception as exc:
            result["alignment_warning"] = str(exc)
    else:
        raise ValueError("Unsupported Whisper backend")
    result["provenance"] = runtime(config, ["faster-whisper", "ctranslate2", "whisperx", "torch"])
    out.write_text(json.dumps(plain(result), ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["mmpose", "whisper", "check"])
    parser.add_argument("--input")
    parser.add_argument("--config")
    parser.add_argument("--out")
    parser.add_argument("--module")
    args = parser.parse_args()
    if args.kind == "check":
        try:
            found = importlib.util.find_spec(args.module) is not None
        except Exception:
            found = False
        print(json.dumps({"available": found, "python": sys.version.split()[0]}))
        return
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))[args.kind]
    if args.kind == "mmpose":
        pose(Path(args.input), config, Path(args.out))
    else:
        speech(Path(args.input), config, Path(args.out))


if __name__ == "__main__":
    main()
