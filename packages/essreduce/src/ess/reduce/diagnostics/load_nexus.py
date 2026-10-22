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

from ..nexus._nexus_loader import _unique_child_group, read_nexus_file_info
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

AnyMonitor = NewType("AnyMonitor", int)


def _make_workflow(
    filename: str | Path, bank_sizes: DetectorBankSizes | None = None
) -> sl.Pipeline:
    wf = GenericNeXusWorkflow(run_types=[AnyRun], monitor_types=[AnyMonitor])
    wf[Filename[AnyRun]] = filename
    if bank_sizes is not None:
        wf[DetectorBankSizes] = bank_sizes
    return wf


def _find_data_key(entry: snx.Group, group_name: str, filename: str | Path) -> str:
    keys = list(entry[snx.NXevent_data].keys()) + list(entry[snx.NXdata].keys())
    if not keys:
        raise (ValueError(f"No data found in '{group_name}' of file '{filename}'"))
    if len(keys) > 1:
        raise (
            ValueError(
                f"Multiple data found in '{group_name}' of file '{filename}': {keys}"
            )
        )
    return keys[0]


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
    filename: str | Path, path: str, bank_sizes: dict | None = None
) -> sc.DataArray:
    with snx.File(filename) as f:
        entry = _unique_child_group(f, snx.NXentry, None)
        instrument = _unique_child_group(entry, snx.NXinstrument, None)
        group = instrument[path]
        key = _find_data_key(entry=group, group_name=path, filename=filename)
        da = snx.compute_positions(group[()])[key]
    if bank_sizes is not None:
        if da.ndim == 1:
            da = da.fold(dim=da.dim, sizes=bank_sizes)
        else:
            warnings.warn(
                f"Could not reshape data array '{path}' with bank_sizes {bank_sizes} "
                f"because it has more than 1 dimension",
                UserWarning,
                stacklevel=2,
            )
    return da


def _robust_load(
    filename: str | Path,
    kind: Literal[snx.NXdetector, snx.NXmonitor],
    entries: list[str] | str | None = None,
    bank_sizes: DetectorBankSizes | None = None,
) -> sc.DataGroup:

    if not Path(filename).exists():
        raise FileNotFoundError(f"File '{filename}' does not exist")

    user_requested = None
    if entries is None:
        user_requested = []
        file_info = read_nexus_file_info(filename)
        if kind == snx.NXdetector:
            entries = list(file_info.detectors.keys())
        elif kind == snx.NXmonitor:
            entries = list(file_info.monitors.keys())
    if isinstance(entries, str):
        entries = [entries]
    if user_requested is None:
        user_requested = entries

    wf = _make_workflow(filename, bank_sizes=bank_sizes)

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
                    bank_sizes=bank_sizes.get(item) if bank_sizes is not None else None,
                )
            except Exception:  # noqa: S110
                pass

        if item not in dg:
            msg = f"Failed to load {item}"
            if item in user_requested:
                # If the item was explicitly requested by the user, raise an error
                raise ValueError(msg)
            else:
                # Otherwise, issue a warning instead
                warnings.warn(msg, UserWarning, stacklevel=2)
        elif isinstance(dg[item], sc.DataGroup):
            warnings.warn(
                f"{item} could not be loaded as a DataArray; loaded it as a DataGroup "
                "as a fallback",
                UserWarning,
                stacklevel=2,
            )

    return dg


def load_detectors(
    filename: str | Path,
    *,
    detectors: list[str] | str | None = None,
    bank_sizes: DetectorBankSizes | None = None,
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
    bank_sizes:
        Dictionary specifying how to reshape each detector's data array. The keys should
        correspond to detector names, and the values should be dictionaries mapping
        dimension names to reshaped sizes. If ``None``, no reshaping is applied.
    """

    return _robust_load(
        filename, kind=snx.NXdetector, entries=detectors, bank_sizes=bank_sizes
    )


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
