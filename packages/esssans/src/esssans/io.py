# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from .types import (
    Filename,
    RunType,
    RawData,
    MonitorType,
    RawMonitor,
    NeXusMonitorName,
    DetectorEdgeMask,
    SampleHolderMask,
    SampleRun,
    MaskedData,
)


def load(filename: Filename[RunType]) -> RawData[RunType]:
    return RawData[RunType](sc.io.load_hdf5(filename=filename))


def get_monitor(
    da: RawData[RunType], nexus_name: NeXusMonitorName[MonitorType]
) -> RawMonitor[RunType, MonitorType]:
    return RawMonitor(da.attrs[nexus_name].value)
    # TODO We get an exception about __init__ when using this with a DataArray:
    return IncidentMonitor[RunType](da.attrs['monitor2'].value)


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
    da: RawData[RunType], edge_mask: DetectorEdgeMask, holder_mask: SampleHolderMask
) -> MaskedData[RunType]:
    da = da.copy(deep=False)
    da.masks['edges'] = edge_mask
    da.masks['holder_mask'] = holder_mask
    return MaskedData[RunType](da)


providers = [load, get_monitor, detector_edge_mask, sample_holder_mask, mask_detectors]
