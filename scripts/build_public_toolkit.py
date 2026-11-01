#!/usr/bin/env python3
"""Build an explicit source-only release candidate; never traverse local data."""

import argparse
import hashlib
import json
import re
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ["README.md", ".gitignore", ".gitattributes", "config.example.ini", "run_pipeline.py", "adualnet.py",
         "requirements-toolkit.txt", "requirements-video.txt", "requirements-basic.txt", "requirements-paper.txt", "requirements-whisperx.txt",
         "recognition.example.json", "recognition.ssh.example.json",
         "Launch_ADualNet.command", "Launch_ADualNet.bat", "LICENSE_NOTICE.md",
         "docs/toolkit_ui_preview.png", "docs/toolkit_ui_mobile.png", "docs/toolkit_video_review_preview.png", "docs/toolkit_ui_english.png",
         "docs/toolkit_medical_workspace.png", "docs/toolkit_medical_review.png", "public_demo/.nojekyll"]
DIRECTORIES = ["adualnet_toolkit", "scripts", "src", "docs", "tests", "examples", "data/templates", "public_demo", ".github/workflows"]
EXTENSIONS = {".py", ".js", ".css", ".md", ".json", ".html", ".csv", ".xlsx", ".yaml", ".yml"}


def source_files(root=ROOT):
    files = [root / name for name in FILES if (root / name).is_file()]
    if (root / "LICENSE").is_file():
        files.append(root / "LICENSE")
    for directory in DIRECTORIES:
        files.extend(p for p in (root / directory).rglob("*") if p.is_file() and
                     p.suffix in EXTENSIONS and "__pycache__" not in p.parts and not p.is_symlink())
    return sorted(set(files))


def has_private_home_path(raw):
    # Prefix-only regex assertions are not user paths; named home directories are.
    return bool(re.search(rb"(?:/Users|/home)/[A-Za-z0-9._-]+", raw))


def build(out):
    out = Path(out)
    if out.exists():
        raise ValueError("Choose a new ZIP filename; existing release candidates are preserved.")
    files = source_files()
    inventory = []
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f"Source file escapes release root: {path.relative_to(ROOT)}")
        raw = path.read_bytes()
        if has_private_home_path(raw):
            raise ValueError(f"Private home path in {path.relative_to(ROOT)}")
        inventory.append({"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, "adualnet/" + path.relative_to(ROOT).as_posix())
        archive.writestr("adualnet/RELEASE_MANIFEST.json", json.dumps({
            "release": "0.4.0", "source_only": True,
            "license_selected": (ROOT / "LICENSE").exists(), "files": inventory,
            "excluded": ["git history", "participant data", "local workspace", "results", "model weights", "caches"],
        }, indent=2))
    print(f"[DONE] Source-only release candidate: {out} ({len(files)} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "dist/adualnet-toolkit-0.4.0.zip"))
    args = parser.parse_args()
    build(args.out)
