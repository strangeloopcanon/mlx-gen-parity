#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def read_version_pyinit(path: Path) -> str:
    s = path.read_text()
    m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", s)
    if not m:
        raise SystemExit("Could not find __version__ in __init__.py")
    return m.group(1)


def write_version_pyinit(path: Path, version: str) -> None:
    s = path.read_text()
    s = re.sub(r"__version__\s*=\s*['\"][^'\"]+['\"]", f"__version__ = \"{version}\"", s)
    path.write_text(s)


def read_version_pyproject(path: Path) -> str:
    s = path.read_text()
    m = re.search(r"^version\s*=\s*['\"]([^'\"]+)['\"]", s, flags=re.M)
    if not m:
        raise SystemExit("Could not find version in pyproject.toml")
    return m.group(1)


def write_version_pyproject(path: Path, version: str) -> None:
    s = path.read_text()
    s = re.sub(r"^version\s*=\s*['\"][^'\"]+['\"]", f"version = \"{version}\"", s, flags=re.M)
    path.write_text(s)


def bump(ver: str, part: str) -> str:
    major, minor, patch = [int(x) for x in ver.split(".")]
    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1
    return f"{major}.{minor}.{patch}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bump project version")
    parser.add_argument(
        "part",
        nargs="?",
        default="patch",
        choices=("patch", "minor", "major"),
        help="Which semantic version part to bump (default: patch)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    part = args.part
    init_path = Path("mlx_genkit/__init__.py")
    proj_path = Path("pyproject.toml")
    old = read_version_pyinit(init_path)
    new = bump(old, part)
    write_version_pyinit(init_path, new)
    write_version_pyproject(proj_path, new)
    print(new)


if __name__ == "__main__":
    main()
