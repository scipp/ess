# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Fail-safe loaders for NeXus files
"""

from pathlib import Path
from typing import NewType

import sciline as sl
import scipp as sc
import scippnexus as snx

from .types import (
    AnyRun,
    Filename,
    NeXusDetectorName,
    NeXusName,
    RawDetector,
    RawMonitor,
)
from .workflow import GenericNeXusWorkflow

INSTRUMENT_PATH = '/entry/instrument'

AnyMonitor = NewType("AnyMonitor", int)


def _make_workflow(filename: str | Path) -> sl.Pipeline:
    wf = GenericNeXusWorkflow(run_types=[AnyRun], monitor_types=[AnyMonitor])
    wf[Filename[AnyRun]] = filename
    return wf


def _load_detector_with_workflow(wf: sl.Pipeline, bank: str) -> sc.DataArray:
    wf[NeXusDetectorName] = bank
    return wf.compute(RawDetector[AnyRun])


def _find_data_keys(entry: snx.Group, bank: str, filename: str | Path) -> list[str]:
    keys = list(entry[snx.NXevent_data].keys()) + list(entry[snx.NXdata].keys())
    if not keys:
        raise (ValueError(f"No data found in '{bank}' of file '{filename}'"))
    if len(keys) > 1:
        raise (
            ValueError(f"Multiple data found in '{bank}' of file '{filename}': {keys}")
        )
    return keys


def _load_detector_with_fallback(
    filename: str | Path, bank: str
) -> sc.DataArray | None:
    with snx.File(filename) as f:
        entry = f[f'{INSTRUMENT_PATH}/{bank}']
        keys = _find_data_keys(entry, bank, filename)
        da = snx.compute_positions(entry[()])[keys[0]]
        return da


def load_detectors(
    filename: str | Path,
    banks: list[str] | str,
    fold: dict | None = None,
) -> sc.DataGroup:
    """
    Load detector data from a NeXus file. We attempt to use the GenericNeXusWorkflow
    first and fall back to raw Scippnexus code if necessary.

    Parameters
    ----------
    filename:
        Path to the NeXus file.
    banks:
        List of detector banks to load. A single string can also be provided to load
        only one bank.
    """

    if isinstance(banks, str):
        banks = [banks]

    wf = _make_workflow(filename)

    dg = sc.DataGroup()
    for bank in banks:
        try:
            dg[bank] = _load_detector_with_workflow(wf, bank)
        except Exception:  # noqa: PERF203
            try:
                dg[bank] = _load_detector_with_fallback(filename, bank)
            except Exception:  # noqa: S112
                continue

    if fold is not None:
        for key, da in dg.items():
            if key in fold:
                dg[key] = da.fold(dim=da.dim, sizes=fold[key])
    return dg


def _load_monitor_with_workflow(wf: sl.Pipeline, monitor: str) -> sc.DataArray:
    wf[NeXusName[AnyMonitor]] = monitor
    return wf.compute(RawMonitor[AnyRun, AnyMonitor])


def _load_monitor_with_fallback(filename: str | Path, monitor: str) -> sc.DataArray:
    with snx.File(filename) as f:
        entry = f[f'{INSTRUMENT_PATH}/{monitor}']
        keys = _find_data_keys(entry, monitor, filename)
        da = snx.compute_positions(entry[()])[keys[0]]
    return da


def load_monitors(filename: str | Path, monitors: list[str] | str) -> sc.DataGroup:
    """
    Load monitor data from a NeXus file. We attempt to use the GenericNeXusWorkflow
    first and fall back to raw Scippnexus code if necessary.

    Parameters
    ----------
    filename:
        Path to the NeXus file.
    monitors:
        List of monitors to load. A single string can also be provided to load
        only one monitor.
    """

    if isinstance(monitors, str):
        monitors = [monitors]

    wf = _make_workflow(filename)

    dg = sc.DataGroup()
    for monitor in monitors:
        try:
            dg[monitor] = _load_monitor_with_workflow(wf, monitor)
        except Exception:  # noqa: PERF203
            try:
                dg[monitor] = _load_monitor_with_fallback(filename, monitor)
            except Exception:  # noqa: S112
                continue

    return dg
