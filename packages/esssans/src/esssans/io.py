# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
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


def load(filename: Filename[RunType]) -> RawData[RunType]:
    da = sc.io.load_hdf5(filename=filename)
    if 'gravity' not in da.coords:
        da.coords["gravity"] = gravity_vector()
    return RawData[RunType](da)


def load_direct_beam(filename: DirectBeamFilename) -> DirectBeam:
    return DirectBeam(sc.io.load_hdf5(filename=filename))


def get_monitor(
    da: RawData[RunType], nexus_name: NeXusMonitorName[MonitorType]
) -> RawMonitor[RunType, MonitorType]:
    return RawMonitor[RunType, MonitorType](da.attrs[nexus_name].value)


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
    da = da.copy(deep=False)
    if edge_mask is not None:
        da.masks['edges'] = edge_mask
    if holder_mask is not None:
        da.masks['holder_mask'] = holder_mask
    return MaskedData[RunType](da)


providers = [
    load,
    load_direct_beam,
    get_monitor,
    detector_edge_mask,
    sample_holder_mask,
    mask_detectors,
]
