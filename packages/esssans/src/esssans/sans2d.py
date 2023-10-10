# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Loading and masking specific to the ISIS Sans2d instrument and files stored in Scipp's
HDF5 format.
"""
from typing import Optional

import scipp as sc

from .common import gravity_vector
from .types import (
    DetectorEdgeMask,
    DirectBeam,
    DirectBeamFilename,
    Filename,
    MaskedData,
    MonitorType,
    NeXusMonitorName,
    RawData,
    RawMonitor,
    RunType,
    SampleHolderMask,
    SampleRun,
)


def pooch_load(filename: Filename[RunType]) -> RawData[RunType]:
    from .data import get_path

    da = sc.io.load_hdf5(filename=get_path(filename))
    if 'gravity' not in da.coords:
        da.coords["gravity"] = gravity_vector()
    return RawData[RunType](da)


def pooch_load_direct_beam(filename: DirectBeamFilename) -> DirectBeam:
    from .data import get_path

    return DirectBeam(sc.io.load_hdf5(filename=get_path(filename)))


def get_monitor(
    da: RawData[RunType], nexus_name: NeXusMonitorName[MonitorType]
) -> RawMonitor[RunType, MonitorType]:
    # See https://github.com/scipp/sciline/issues/52 why copy needed
    mon = da.attrs[nexus_name].value.copy()
    return RawMonitor[RunType, MonitorType](mon)


def detector_edge_mask(sample: RawData[SampleRun]) -> DetectorEdgeMask:
    mask_edges = (
        sc.abs(sample.coords['position'].fields.x) > sc.scalar(0.48, unit='m')
    ) | (sc.abs(sample.coords['position'].fields.y) > sc.scalar(0.45, unit='m'))
    return DetectorEdgeMask(mask_edges)


def sample_holder_mask(sample: RawData[SampleRun]) -> SampleHolderMask:
    summed = sample.sum('tof')
    holder_mask = (
        (summed.data < sc.scalar(100, unit='counts'))
        & (sample.coords['position'].fields.x > sc.scalar(0, unit='m'))
        & (sample.coords['position'].fields.x < sc.scalar(0.42, unit='m'))
        & (sample.coords['position'].fields.y < sc.scalar(0.05, unit='m'))
        & (sample.coords['position'].fields.y > sc.scalar(-0.15, unit='m'))
    )
    return SampleHolderMask(holder_mask)


def mask_detectors(
    da: RawData[RunType],
    edge_mask: Optional[DetectorEdgeMask],
    holder_mask: Optional[SampleHolderMask],
) -> MaskedData[RunType]:
    """Apply pixel-specific masks to raw data.

    Parameters
    ----------
    da:
        Raw data.
    edge_mask:
        Mask for detector edges.
    holder_mask:
        Mask for sample holder.
    """
    da = da.copy(deep=False)
    if edge_mask is not None:
        da.masks['edges'] = edge_mask
    if holder_mask is not None:
        da.masks['holder_mask'] = holder_mask
    return MaskedData[RunType](da)


providers = [
    pooch_load_direct_beam,
    pooch_load,
    get_monitor,
    detector_edge_mask,
    sample_holder_mask,
    mask_detectors,
]
"""
Providers for loading and masking Sans2d data.

These are meant for complementing the top-level :py:data:`esssans.providers` list.
"""
