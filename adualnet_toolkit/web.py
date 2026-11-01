"""Loopback-only UI, no external assets/telemetry; optional trusted SSH inference."""

import hmac
import json
import mimetypes
import secrets
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD = 20 * 1024 * 1024


def make_handler(workspace, token, port, recognition_config=None):
    workspace = Path(workspace).resolve()
    inputs, runs = workspace / "inputs", workspace / "runs"
    inputs.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    jobs, uploads, gate = {}, {}, threading.Lock()
    from .video_routes import MAX_VIDEO, VideoService, send_media
    videos = VideoService(workspace, recognition_config)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, content, content_type="application/json", status=200):
            raw = json.dumps(content, ensure_ascii=False).encode() if content_type == "application/json" else content
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(raw)

        def authorized(self):
            host = self.headers.get("Host", "")
            if host not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                return False
            supplied = self.headers.get("X-ADualNet-Token", "") or parse_qs(urlsplit(self.path).query).get("token", [""])[0]
            origin = self.headers.get("Origin")
            return (hmac.compare_digest(supplied, token) and
                    (origin is None or origin in (f"http://127.0.0.1:{port}", f"http://localhost:{port}")))

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/" and self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}"):
                page = (ROOT / "adualnet_toolkit/ui.html").read_text(encoding="utf-8").replace("__LOCAL_TOKEN__", token)
                catalog = (ROOT / "adualnet_toolkit/ui_translations.json").read_text(encoding="utf-8")
                page = page.replace("__UI_TRANSLATIONS__", catalog.replace("<", "\\u003c"))
                self.send(page.encode(), "text/html; charset=utf-8")
                return
            if not self.authorized():
                self.send({"error": "Local session token required."}, status=403)
                return
            try:
                if path == "/assets/workspace.css":
                    self.send((ROOT / "adualnet_toolkit/workspace.css").read_bytes(), "text/css; charset=utf-8")
                    return
                if path in ["/assets/video_ui.js", "/assets/ui_i18n.js", "/assets/workspace_ui.js"]:
                    self.send((ROOT / "adualnet_toolkit" / path.rsplit("/", 1)[-1]).read_bytes(), "text/javascript; charset=utf-8")
                    return
                if path.startswith("/video-media/"):
                    send_media(self, videos.media_path(path))
                    return
                if path.startswith("/api/video/"):
                    value = videos.get_api(path)
                    if value is not None:
                        self.send(value)
                        return
            except Exception as exc:
                self.send({"error": str(exc)}, status=400)
                return
            if path.startswith("/api/job/"):
                job_id = path.split("/")[-1]
                with gate:
                    job = jobs.get(job_id)
                if not job:
                    self.send({"error": "Unknown run"}, status=404)
                    return
                code = job["process"].poll()
                state = "running" if code is None else "complete" if code == 0 and (runs / job_id / "COMPLETE.json").is_file() else "failed"
                log = job["log"].read_text(errors="replace")[-16000:]
                self.send({"state": state, "log": log, "report": f"/runs/{job_id}/report.html?token={token}" if state == "complete" else None})
                return
            parts = path.strip("/").split("/")
            allowed = {"report.html", "overview.svg", "overview.png", "performance.csv", "paired_delta_auroc.csv", "analysis_tables.xlsx", "calibration_summary.csv", "protocol.json"}
            if len(parts) == 3 and parts[0] == "runs" and parts[1] in jobs and parts[2] in allowed:
                file = runs / parts[1] / parts[2]
                if file.is_file():
                    self.send(file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")
                    return
            self.send({"error": "Not found"}, status=404)

        def do_POST(self):
            if not self.authorized():
                self.send({"error": "Invalid local session"}, status=403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                path = urlsplit(self.path).path
                if path == "/api/video/upload":
                    self.send(videos.upload(self.rfile, length, self.headers.get("X-File-Type", "").lower()))
                    return
                if length < 1 or length > MAX_UPLOAD:
                    raise ValueError("Use a nonempty file smaller than 20 MB.")
                payload = self.rfile.read(length)
                path = urlsplit(self.path).path
                if path == "/api/upload":
                    suffix = self.headers.get("X-File-Type", "").lower()
                    if suffix not in (".csv", ".xlsx"):
                        raise ValueError("Upload CSV or XLSX.")
                    file_id = uuid.uuid4().hex
                    target = inputs / (file_id + suffix)
                    target.write_bytes(payload)
                    with gate:
                        uploads[file_id] = target
                    self.send({"upload_id": file_id})
                    return
                config = json.loads(payload)
                if path.startswith("/api/video/"):
                    with gate:
                        value = videos.post_api(path, config)
                    if value is not None:
                        self.send(value)
                        return
                if path == "/api/validate":
                    from .analysis import load_table, prepare_table
                    with gate:
                        target = uploads.get(config.get("upload_id"))
                    if not target:
                        raise ValueError("Upload a feature table first.")
                    _, _, report = prepare_table(load_table(target, str(config.get("sheet", "0"))))
                    self.send(report)
                    return
                if path == "/api/run":
                    with gate:
                        if any(job["process"].poll() is None for job in jobs.values()):
                            raise ValueError("A run is already active. Wait for it to finish.")
                        run_id = uuid.uuid4().hex
                        out = runs / run_id
                        base = [sys.executable, "-u", str(ROOT / "adualnet.py")]
                        if config.get("demo") is True:
                            command = base + ["demo", "--out", str(out)]
                        else:
                            target = uploads.get(config.get("upload_id"))
                            if not target:
                                raise ValueError("Upload a feature table first.")
                            version = str(config.get("measurement_version", "")).strip()
                            outcome = str(config.get("outcome_definition", "")).strip()
                            if not version or not outcome:
                                raise ValueError("Fill in the measurement version and outcome definition.")
                            command = base + ["analyze", "--input", str(target), "--out", str(out), "--sheet", str(config.get("sheet", "0")),
                                              "--measurement-version", version, "--outcome-definition", outcome]
                        log_path = runs / (run_id + ".log")
                        with log_path.open("wb") as log_file:
                            proc = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
                        jobs[run_id] = {"process": proc, "log": log_path}
                    self.send({"job_id": run_id})
                    return
                self.send({"error": "Not found"}, status=404)
            except Exception as exc:
                self.send({"error": str(exc)}, status=400)

        def do_HEAD(self):
            path = urlsplit(self.path).path
            if not self.authorized():
                self.send_response(403)
                self.end_headers()
                return
            try:
                file = videos.media_path(path)
                if file is None:
                    raise ValueError("Unknown media route")
                send_media(self, file, head=True)
            except Exception:
                self.send_response(404)
                self.end_headers()

    Handler.jobs = jobs
    return Handler


def serve_local(workspace, port, recognition_config=None):
    if not 1024 <= port <= 65535:
        raise ValueError("Choose a port from 1024 to 65535.")
    token = secrets.token_urlsafe(32)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(workspace, token, port, recognition_config))
    print(f"[INFO] ADualNet local interface: http://127.0.0.1:{port}", flush=True)
    print(f"[INFO] Local workspace: {Path(workspace).resolve()}; Ctrl+C stops the interface.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[DONE] Interface stopped. Running analysis subprocesses retain their logs.")
    finally:
        server.server_close()
