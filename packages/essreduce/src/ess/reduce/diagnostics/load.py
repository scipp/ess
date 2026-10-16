# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Fail-safe loaders for NeXus files
"""

import warnings
from pathlib import Path
from typing import Literal, NewType

import sciline as sl
import scipp as sc
import scippnexus as snx

from ..nexus.types import (
    AnyRun,
    DetectorBankSizes,
    Filename,
    NeXusDetectorName,
    NeXusName,
    RawDetector,
    RawMonitor,
)
from ..nexus.workflow import GenericNeXusWorkflow

INSTRUMENT_PATH = '/entry/instrument'

AnyMonitor = NewType("AnyMonitor", int)


def _make_workflow(filename: str | Path, fold: dict | None = None) -> sl.Pipeline:
    wf = GenericNeXusWorkflow(run_types=[AnyRun], monitor_types=[AnyMonitor])
    wf[Filename[AnyRun]] = filename
    if fold is not None:
        wf[DetectorBankSizes] = fold
    return wf


def _find_detectors_or_monitors(
    filename: str | Path, kind: Literal[snx.NXdetector, snx.NXmonitor]
) -> list[str]:
    with snx.File(filename) as f:
        items = f[INSTRUMENT_PATH][kind]
    return list(items.keys())


def _find_data_keys(
    entry: snx.Group, group_name: str, filename: str | Path
) -> list[str]:
    keys = list(entry[snx.NXevent_data].keys()) + list(entry[snx.NXdata].keys())
    if not keys:
        raise (ValueError(f"No data found in '{group_name}' of file '{filename}'"))
    if len(keys) > 1:
        raise (
            ValueError(
                f"Multiple data found in '{group_name}' of file '{filename}': {keys}"
            )
        )
    return keys


def _load_with_workflow(
    wf: sl.Pipeline, path: str, kind: Literal[snx.NXdetector, snx.NXmonitor]
) -> sc.DataArray:
    if kind == snx.NXdetector:
        wf[NeXusDetectorName] = path
        return wf.compute(RawDetector[AnyRun])
    if kind == snx.NXmonitor:
        wf[NeXusName[AnyMonitor]] = path
        return wf.compute(RawMonitor[AnyRun, AnyMonitor])
    raise ValueError(
        f"Unsupported kind when loading with workflow: {kind}. "
        f"Expected one of: {snx.NXdetector}, {snx.NXmonitor}."
    )


def _load_with_fallback(
    filename: str | Path, path: str, fold: dict | None = None
) -> sc.DataArray:
    with snx.File(filename) as f:
        entry = f[f'{INSTRUMENT_PATH}/{path}']
        keys = _find_data_keys(entry=entry, group_name=path, filename=filename)
        da = snx.compute_positions(entry[()])[keys[0]]
    if fold is not None:
        if da.ndim == 1:
            da = da.fold(dim=da.dim, sizes=fold)
        else:
            warnings.warn(
                f"Could not fold data array '{path}' with fold {fold} because it has "
                "more than 1 dimension",
                UserWarning,
                stacklevel=2,
            )
    return da


def _robust_load(
    filename: str | Path,
    kind: Literal[snx.NXdetector, snx.NXmonitor],
    entries: list[str] | str | None = None,
    fold: dict | None = None,
) -> sc.DataGroup:

    user_requested = None
    if entries is None:
        user_requested = []
        entries = _find_detectors_or_monitors(filename, kind)
    if isinstance(entries, str):
        entries = [entries]
    if user_requested is None:
        user_requested = entries

    wf = _make_workflow(filename, fold=fold)

    dg = sc.DataGroup()
    for item in entries:
        try:
            dg[item] = _load_with_workflow(wf, path=item, kind=kind)
        except Exception:
            warnings.warn(
                f"{item}: loading with default loader failed, attempting fallback",
                UserWarning,
                stacklevel=2,
            )
            try:
                dg[item] = _load_with_fallback(
                    filename,
                    path=item,
                    fold=fold.get(item) if fold is not None else None,
                )
            except Exception:  # noqa: S112
                continue

        if item not in dg:
            msg = f"Failed to load {item}"
            if item in user_requested:
                # If the item was explicitly requested by the user, raise an error
                raise RuntimeError(msg)
            else:
                # Otherwise, issue a warning instead
                warnings.warn(msg, UserWarning, stacklevel=2)
        elif isinstance(dg[item], sc.DataGroup):
            warnings.warn(
                f"{item} could not be loaded as a DataArray, falled back to "
                "loading it as a DataGroup",
                UserWarning,
                stacklevel=2,
            )

    return dg


def load_detectors(
    filename: str | Path,
    *,
    detectors: list[str] | str | None = None,
    fold: dict | None = None,
) -> sc.DataGroup:
    """
    Load detector data from a NeXus file. We attempt to use the GenericNeXusWorkflow
    first and fall back to raw Scippnexus code if necessary.
    By default, an attempt at loading each detector as a DataArray is made. If this
    fails (because signal shapes are incompatible or some fields are missing),
    the loader will fall back to loading the detector as a DataGroup.

    Parameters
    ----------
    filename:
        Path to the NeXus file.
    detectors:
        List of detectors to load. A single string can also be provided to load
        only one detector. If ``None``, all NXdetectors are loaded.
    fold:
        Dictionary specifying how to fold each detector's data array. The keys should
        correspond to detector names, and the values should be dictionaries mapping
        dimension names to fold sizes. If ``None``, no folding is applied.
    """

    return _robust_load(filename, kind=snx.NXdetector, entries=detectors, fold=fold)


def load_monitors(
    filename: str | Path, *, monitors: list[str] | str | None = None
) -> sc.DataGroup:
    """
    Load monitor data from a NeXus file. We attempt to use the GenericNeXusWorkflow
    first and fall back to raw Scippnexus code if necessary.
    By default, an attempt at loading each monitor as a DataArray is made. If this
    fails (because signal shapes are incompatible or some fields are missing),
    the loader will fall back to loading the monitor as a DataGroup.

    Parameters
    ----------
    filename:
        Path to the NeXus file.
    monitors:
        List of monitors to load. A single string can also be provided to load
        only one monitor. If ``None``, all NXmonitors are loaded.
    """

    return _robust_load(filename, kind=snx.NXmonitor, entries=monitors)
