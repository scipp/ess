# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Fail-safe loaders for loki files
"""

from pathlib import Path

import sciline as sl
import scipp as sc
import scippnexus as snx

from ..sans.types import (
    BeamCenter,
    Filename,
    Incident,
    NeXusDetectorName,
    NeXusMonitorName,
    RawDetector,
    RawMonitor,
    SampleRun,
)
from .workflow import DETECTOR_BANK_SIZES, LokiWorkflow


def _make_workflow(filename: str | Path) -> sl.Pipeline:
    # Create workflow to try and load the data properly first
    wf = LokiWorkflow()
    wf[BeamCenter] = sc.vector([0, 0, 0], unit='m')
    wf[Filename[SampleRun]] = filename
    return wf


def _load_detector_with_workflow(wf: sl.Pipeline, bank: str) -> sc.DataArray:
    wf[NeXusDetectorName] = bank
    return wf.compute(RawDetector[SampleRun])


def _load_detector_with_fallback(filename: str | Path, bank: str) -> sc.DataArray:
    with snx.File(filename) as f:
        da = snx.compute_positions(f[f'/entry/instrument/{bank}'][()])[
            "detector_events"
        ]

        # Bank 0 is mounted on a movable stage, and has a time-dependent
        # NXtransformation. If the transformation log is not populated,
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
        return da


def load_detectors(
    filename: str | Path, banks: list[str] | str | None = None
) -> sc.DataGroup:
    """
    Load detector data from a Loki file. We attempt to use the LokiWorkflow first and
    fall back to raw Scippnexus code if necessary.

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

    wf = _make_workflow(filename)

    dg = sc.DataGroup()
    for bank in banks:
        try:
            dg[bank] = _load_detector_with_workflow(wf, bank)
        except Exception:  # noqa: PERF203, RUF100, S112
            try:
                dg[bank] = _load_detector_with_fallback(filename, bank)
            except Exception:  # noqa: PERF203, RUF100, S112
                continue

    return dg


def _load_monitor_with_workflow(wf: sl.Pipeline, monitor: str) -> sc.DataArray:
    wf[NeXusMonitorName[Incident]] = monitor
    return wf.compute(RawMonitor[SampleRun, Incident])


def _load_monitor_with_fallback(filename: str | Path, monitor: str) -> sc.DataArray:
    with snx.File(filename) as f:
        da = snx.compute_positions(f[f'/entry/instrument/{monitor}'][()])[
            "monitor_events"
        ]
    return da


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

    wf = _make_workflow(filename)

    dg = sc.DataGroup()
    for monitor in monitors:
        try:
            dg[monitor] = _load_monitor_with_workflow(wf, monitor)
        except Exception:  # noqa: PERF203, RUF100, S112
            try:
                dg[monitor] = _load_monitor_with_fallback(filename, monitor)
            except Exception:  # noqa: PERF203, RUF100, S112
                continue

    return dg
