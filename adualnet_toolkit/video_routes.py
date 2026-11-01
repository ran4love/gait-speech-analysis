"""Local video jobs and allow-listed, seekable media responses."""

import mimetypes
import re
import subprocess
import sys
import uuid
from pathlib import Path

from .video import ROOT, current_review, export_features, load_profile, preflight, read, save_review

MAX_VIDEO = 512 * 1024 * 1024


class VideoService:
    def __init__(self, workspace, profile_path=None):
        self.root = Path(workspace) / "video_jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.profile_path = Path(profile_path or ROOT / "recognition.example.json").resolve()
        self.processes = {}

    def folder(self, identifier):
        if not re.fullmatch(r"[0-9a-f]{32}", identifier):
            raise ValueError("Invalid video job ID.")
        folder = self.root / identifier
        if not folder.is_dir():
            raise ValueError("Unknown video job.")
        return folder

    def upload(self, stream, length, suffix):
        if suffix not in [".mp4", ".mov", ".mkv", ".avi", ".webm"] or not 0 < length <= MAX_VIDEO:
            raise ValueError("Choose an MP4/MOV/MKV/AVI/WebM video up to 512 MB.")
        identifier = uuid.uuid4().hex
        folder = self.root / identifier
        folder.mkdir()
        with (folder / ("source" + suffix)).open("xb") as output:
            remaining = length
            while remaining:
                part = stream.read(min(remaining, 1024 * 1024))
                if not part:
                    raise ValueError("Video upload was interrupted. Upload it again as a new job.")
                output.write(part)
                remaining -= len(part)
        return {"job_id": identifier}

    def launch(self, identifier=None, demo=False):
        if any(p.poll() is None for p in self.processes.values()):
            raise ValueError("A video recognition job is already active.")
        if demo:
            identifier = uuid.uuid4().hex
            (self.root / identifier).mkdir()
        folder = self.folder(identifier)
        if (folder / "state.json").exists():
            raise ValueError("This video already has a run. Upload a new job to rerun recognition.")
        cmd = [sys.executable, "-u", str(ROOT / "adualnet.py"), "video-demo" if demo else "video",
               "--out", str(folder), "--recognition-config", str(self.profile_path)]
        if not demo:
            sources = list(folder.glob("source.*"))
            if len(sources) != 1:
                raise ValueError("Upload one source video before starting.")
            cmd += ["--input", str(sources[0])]
        with (folder / "run.log").open("xb") as log:
            self.processes[identifier] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        return {"job_id": identifier}

    def status(self, identifier):
        folder = self.folder(identifier)
        status = read(folder / "state.json") if (folder / "state.json").is_file() else {"status": "running", "message": "启动视频流程"}
        process = self.processes.get(identifier)
        if process is not None and process.poll() not in (None, 0) and status["status"] == "running":
            status = {"status": "failed", "message": "Worker exited; inspect run.log."}
        status["job_id"] = identifier
        status["log"] = (folder / "run.log").read_text(errors="replace")[-6000:] if (folder / "run.log").exists() else ""
        if status["status"] in ["awaiting_review", "reviewed", "features_ready"]:
            status["media"] = read(folder / "media.json")
            status["tracks"] = read(folder / "tracks.json")
            status["review"] = current_review(folder)
        return status

    def get_api(self, path):
        if path == "/api/video/preflight":
            return preflight(load_profile(self.profile_path))
        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[:2] == ["api", "video"]:
            return self.status(parts[2])
        return None

    def post_api(self, path, payload):
        if path == "/api/video/demo":
            return self.launch(demo=True)
        parts = path.strip("/").split("/")
        if len(parts) != 4 or parts[:2] != ["api", "video"]:
            return None
        identifier, action = parts[2:]
        folder = self.folder(identifier)
        if action == "start":
            if payload.get("processing_authorized") is not True:
                raise ValueError("Confirm authorized processing/transfer and retention before recognition.")
            return self.launch(identifier)
        if action == "review":
            return {"review": save_review(folder, payload)}
        if action == "export":
            if payload.get("revision") != current_review(folder)["revision"]:
                raise ValueError("Reload the current saved review before exporting.")
            out = export_features(folder)
            return {"revision": current_review(folder)["revision"], "files": [out.name + "/" + p.name for p in out.iterdir() if p.name in ["reviewed_features.xlsx", "reviewed_features.csv", "feature_values.csv", "gait_qc.json", "provenance.json"]]}
        return None

    def media_path(self, path):
        parts = path.strip("/").split("/")
        if len(parts) < 3 or parts[0] != "video-media":
            return None
        folder = self.folder(parts[1])
        if len(parts) == 3 and parts[2] in ["review.mp4", "poses.json", "transcript_raw.json"]:
            file = folder / parts[2]
        elif len(parts) == 4 and re.fullmatch(r"revision_[0-9]{4,}", parts[2]) and parts[3] in ["reviewed_features.xlsx", "reviewed_features.csv", "feature_values.csv", "gait_qc.json", "provenance.json"]:
            file = folder / "features" / parts[2] / parts[3]
        else:
            raise ValueError("Media/output file is not available through this route.")
        if not file.is_file():
            raise ValueError("Requested file is not ready.")
        return file


def send_media(handler, file, head=False):
    """Byte-range support is required for seeking to a response in long videos."""
    size = file.stat().st_size
    start, end, code = 0, size-1, 200
    requested = handler.headers.get("Range")
    if requested:
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
        if not match or not any(match.groups()):
            handler.send({"error": "Unsupported byte range"}, status=416)
            return
        first, last = match.groups()
        if first:
            start, end = int(first), min(int(last), end) if last else end
        else:
            start = max(0, size-int(last))
        if not 0 <= start <= end < size:
            handler.send_response(416)
            handler.send_header("Content-Range", "bytes */{}".format(size))
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return
        code = 206
    handler.send_response(code)
    handler.send_header("Content-Type", mimetypes.guess_type(file.name)[0] or "application/octet-stream")
    handler.send_header("Content-Length", str(end-start+1))
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Referrer-Policy", "no-referrer")
    if code == 206:
        handler.send_header("Content-Range", "bytes {}-{}/{}".format(start, end, size))
    handler.end_headers()
    if head:
        return
    try:
        with file.open("rb") as stream:
            stream.seek(start)
            left = end-start+1
            while left:
                chunk = stream.read(min(left, 256*1024))
                if not chunk:
                    break
                handler.wfile.write(chunk)
                left -= len(chunk)
    except (BrokenPipeError, ConnectionResetError):
        pass  # Normal when the browser seeks to another playback position.
