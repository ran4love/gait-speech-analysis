"""SSH transport for pretrained video/audio inference; no shell credentials."""

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath


def connection(profile):
    settings = profile.get("execution", {})
    argv = settings.get("ssh_argv")
    if not isinstance(argv, list) or len(argv) < 2 or any(not isinstance(a, str) or "\x00" in a for a in argv):
        raise ValueError("SSH mode needs execution.ssh_argv, e.g. ['ssh', 'research-server'].")
    root = settings.get("remote_root", ".cache/adualnet/video_jobs")
    if not isinstance(root, str) or ".." in PurePosixPath(root).parts or not re.fullmatch(r"[a-zA-Z0-9_./-]+", root):
        raise ValueError("Use a simple remote_root directory without '..' or shell characters.")
    return argv, root


def remote_checks(profile):
    ssh, _ = connection(profile)
    checks = []
    for section, module in [("mmpose", "mmpose"), ("whisper", "faster_whisper" if profile["whisper"]["backend"] == "faster-whisper" else "whisperx")]:
        code = "import importlib.util,json,sys;print(json.dumps({'available':importlib.util.find_spec(%r) is not None,'python':sys.version.split()[0]}))" % module
        try:
            result = subprocess.run(ssh + [shlex.join([profile[section]["python"], "-c", code])], capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout.strip())
            ready = result.returncode == 0 and data["available"]
            message = "Ubuntu SSH / Python {} / {}".format(data["python"], module)
        except Exception as exc:
            ready, message = False, str(exc)
        checks.append({"component": section, "available": ready, "message": message})
    return checks


def infer_remote(job, profile):
    from .video import command, dump
    ssh, root = connection(profile)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", job.name):
        raise ValueError("Remote job directory names must be alphanumeric with underscores/hyphens.")
    remote = PurePosixPath(root) / job.name
    command(ssh + [shlex.join(["mkdir", "-p", "--", root]) + " && " + shlex.join(["mkdir", "--", str(remote)])])
    # Only recognition settings are sent; local SSH options and key paths stay local.
    config_file = job / "remote_model_config.json"
    dump(config_file, {"mmpose": profile["mmpose"], "whisper": profile["whisper"]})
    for source, name in [(job / "recognition_worker.py", "recognition_worker.py"), (config_file, "config.json"), (job / "review.mp4", "review.mp4"), (job / "audio.wav", "audio.wav")]:
        print("[RUN] Upload {} to configured Ubuntu inference host".format(name), flush=True)
        with source.open("rb") as stream:
            result = subprocess.run(ssh + ["cat > " + shlex.quote(str(remote / name))], stdin=stream)
            if result.returncode:
                raise RuntimeError("SSH input upload failed.")
    for section, source, target in [("mmpose", "review.mp4", "pose_raw.json"), ("whisper", "audio.wav", "transcript_raw.json")]:
        print("[RUN] Ubuntu {} inference".format(section), flush=True)
        args = [profile[section]["python"], "-u", str(remote / "recognition_worker.py"), section,
                "--input", str(remote / source), "--config", str(remote / "config.json"), "--out", str(remote / target)]
        command(ssh + [shlex.join(args)])
        with (job / target).open("xb") as stream:
            result = subprocess.run(ssh + [shlex.join(["cat", "--", str(remote / target)])], stdout=stream)
            if result.returncode:
                raise RuntimeError("SSH result download failed.")
    dump(job / "remote_receipt.json", {"mode": "ssh", "remote_job_directory": str(remote),
          "transferred_inputs": ["normalized review.mp4", "16kHz mono audio.wav", "recognition adapter", "model settings"],
          "retention": "local and remote copies retained; no automatic deletion",
          "classifier_fitted_or_scored": False})
