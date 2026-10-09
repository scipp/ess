# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Fail-safe loaders for loki files
"""

from pathlib import Path

import scipp as sc

from ess.reduce.nexus import safe_load

from .workflow import DETECTOR_BANK_SIZES


def load_detectors(
    filename: str | Path, detectors: list[str] | str | None = None
) -> sc.DataGroup:
    """
    Robust loader for detector data from a Loki file.

    Parameters
    ----------
    filename:
        Path to the Loki file.
    detectors:
        List of detectors to load. A single string can also be provided to load
        only one detector. If ``None``, all detectors are loaded.
    """
    return safe_load.load_detectors(filename, detectors, fold=DETECTOR_BANK_SIZES)


load_monitors = safe_load.load_monitors
"""Robust loader for monitor data from a Loki file."""
