# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Fail-safe loaders for loki files
"""

from pathlib import Path

import scipp as sc
import scippnexus as snx

from .workflow import DETECTOR_BANK_SIZES


def load_detectors(
    filename: str | Path, banks: list[str] | str | None = None
) -> sc.DataGroup:
    """
    Load detector data from a Loki file.

    Parameters
    ----------
    filename:
        Path to the Loki file.
    banks:
        List of detector banks to load. A single string can also be provided to load
        only one bank. If ``None``, all banks are loaded.
    """

    if banks is None:
        banks = list(DETECTOR_BANK_SIZES.keys())
    if isinstance(banks, str):
        banks = [banks]

    dg = sc.DataGroup()
    with snx.File(filename) as f:
        for bank in banks:
            try:
                da = snx.compute_positions(f[f'/entry/instrument/{bank}'][()])[
                    "detector_events"
                ]

                # Bank 0 is mounted on a movable stage, and has a time-dependent
                # NXtransformation. It the transformation log is not populated,
                # compute_positions fails to yield positions for the pixels. If it is
                # missing, we just assume 0 translation and use the pixel offsets as
                # positions.
                if "position" not in da.coords:
                    da.coords["position"] = sc.spatial.as_vectors(
                        da.coords["x_pixel_offset"],
                        da.coords["y_pixel_offset"],
                        da.coords["z_pixel_offset"],
                    )
                if bank in DETECTOR_BANK_SIZES:
                    da = da.fold(dim="detector_number", sizes=DETECTOR_BANK_SIZES[bank])
                dg[bank] = da
            except Exception:  # noqa: PERF203, S112
                continue

    return dg


def load_monitors(
    filename: str | Path, monitors: list[str] | str | None = None
) -> sc.DataGroup:
    """
    Load monitor data from a Loki file.

    Parameters
    ----------
    filename:
        Path to the Loki file.
    monitors:
        List of monitors to load. A single string can also be provided to load
        only one monitor. If ``None``, all monitors are loaded.
    """

    if monitors is None:
        monitors = [f"beam_monitor_m{i}" for i in range(5)]
    if isinstance(monitors, str):
        monitors = [monitors]

    dg = sc.DataGroup()
    with snx.File(filename) as f:
        for monitor in monitors:
            try:
                da = snx.compute_positions(f[f'/entry/instrument/{monitor}'][()])[
                    "monitor_events"
                ]
                dg[monitor] = da
            except Exception:  # noqa: PERF203, S112
                continue

    return dg
