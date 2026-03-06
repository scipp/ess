#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) Scipp contributors (https://github.com/scipp)
"""Sync requires-python with pixi.toml.

The Python minor version pinned in pixi.toml (e.g. ``python = "3.12.*"``)
is the single source of truth. This script updates:

* ``requires-python`` in ``packages/*/pyproject.toml``
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIXI_TOML = ROOT / "pixi.toml"
PACKAGES = ROOT / "packages"


def python_minor_version_from_pixi() -> str:
    text = PIXI_TOML.read_text()
    m = re.search(r'^python\s*=\s*"(\d+\.\d+)\.\*"', text, re.MULTILINE)
    if not m:
        print(f'ERROR: Could not find python = "X.Y.*" in {PIXI_TOML}')
        sys.exit(1)
    return m.group(1)


def sync_requires_python(version: str) -> bool:
    changed = False
    expected = f'requires-python = ">={version}"'
    pattern = re.compile(r'^requires-python\s*=\s*"[^"]*"', re.MULTILINE)
    for pyproject in sorted(PACKAGES.glob("*/pyproject.toml")):
        text = pyproject.read_text()
        m = pattern.search(text)
        if m is None:
            continue
        if m.group(0) != expected:
            new_text = text[: m.start()] + expected + text[m.end() :]
            pyproject.write_text(new_text)
            print(f"Updated {pyproject.relative_to(ROOT)}")
            changed = True
    return changed


def main() -> int:
    version = python_minor_version_from_pixi()
    changed = sync_requires_python(version)
    if changed:
        print(
            f'Files updated to match python = "{version}.*" from pixi.toml. '
            "Please stage the changes."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
