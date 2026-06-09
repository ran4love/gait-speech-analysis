#!/usr/bin/env python3
"""Shared helpers for the public workflow wrappers."""

from __future__ import annotations

import configparser
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, List


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PUBLIC_ROOT / "src"


def load_config(path: str | Path) -> tuple[configparser.ConfigParser, Path]:
    config_path = Path(path).expanduser().resolve()
    parser = configparser.ConfigParser(interpolation=None)
    with config_path.open("r", encoding="utf-8") as f:
        parser.read_file(f)
    return parser, config_path.parent


def get_path(parser: configparser.ConfigParser, base: Path, section: str, key: str, fallback: str = "") -> Path:
    raw = parser.get(section, key, fallback=fallback).strip()
    raw = os.path.expandvars(os.path.expanduser(raw))
    path = Path(raw)
    return path if path.is_absolute() else (base / path).resolve()


def get_list(parser: configparser.ConfigParser, section: str, key: str, fallback: str = "") -> List[str]:
    raw = parser.get(section, key, fallback=fallback)
    parts = []
    for chunk in raw.replace("\n", ",").split(","):
        item = chunk.strip()
        if item:
            parts.append(item)
    return parts


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_command(cmd: Iterable[str], dry_run: bool = False) -> None:
    cmd = [str(x) for x in cmd]
    print(" ".join(shlex.quote(x) for x in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def run_shell(command: str, dry_run: bool = False) -> None:
    print(command)
    if not dry_run:
        subprocess.run(command, shell=True, check=True)
