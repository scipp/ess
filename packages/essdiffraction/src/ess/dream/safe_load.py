# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Fail-safe loaders for Dream files
"""

from pathlib import Path

import scipp as sc

from ess.reduce.nexus import safe_load

from .workflows import DETECTOR_BANK_SIZES


def load_detectors(
    filename: str | Path, banks: list[str] | str | None = None
) -> sc.DataGroup:
    """
    Robust loader for detector data from a Dream file.

    Parameters
    ----------
    filename:
        Path to the Dream file.
    banks:
        List of detector banks to load. A single string can also be provided to load
        only one bank. If ``None``, all banks are loaded.
    """
    if banks is None:
        banks = list(DETECTOR_BANK_SIZES.keys())
    return safe_load.load_detectors(filename, banks, fold=DETECTOR_BANK_SIZES)


def load_monitors(
    filename: str | Path, monitors: list[str] | str | None = None
) -> sc.DataGroup:
    """
    Robust loader for monitor data from a Dream file.

    Parameters
    ----------
    filename:
        Path to the Dream file.
    monitors:
        List of monitors to load. A single string can also be provided to load
        only one monitor. If ``None``, all monitors are loaded.
    """
    if monitors is None:
        monitors = [f"beam_monitor_m{i}" for i in range(5)]
    return safe_load.load_monitors(filename, monitors)
