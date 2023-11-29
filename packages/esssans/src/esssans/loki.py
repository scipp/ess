# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Loading and masking specific to the ISIS Sans2d instrument and files stored in Scipp's
HDF5 format.
"""
from functools import lru_cache, reduce
from typing import NewType, Optional

import sciline
import scipp as sc
import scippneutron as scn

from .common import gravity_vector
from .types import (
    BackgroundRun,
    CalibratedMaskedData,
    CleanMasked,
    DataWithLogicalDims,
    EmptyBeamRun,
    Filename,
    MaskedData,
    MonitorType,
    NeXusMonitorName,
    Numerator,
    RawData,
    RawMonitor,
    RunType,
    SampleRun,
    SampleRunID,
    TransmissionRun,
    UnmergedSampleRawData,
)

DetectorLowCountsStrawMask = NewType('DetectorLowCountsStrawMask', sc.Variable)
"""Detector low-counts straw mask"""
DetectorBadStrawsMask = NewType('DetectorBadStrawsMask', sc.Variable)
"""Detector bad straws mask"""
DetectorBeamStopMask = NewType('DetectorBeamStopMask', sc.Variable)
"""Detector beam stop mask"""
DetectorTubeEdgeMask = NewType('DetectorTubeEdgeMask', sc.Variable)
"""Detector tube edge mask"""


@lru_cache
def load_loki_run(filename: str) -> sc.DataArray:
    from .data import get_path

    # TODO: Use the new scippnexus to avoid using load_nexus, now that transformations
    # are supported.
    da = scn.load_nexus(get_path(filename))
    if 'gravity' not in da.coords:
        da.coords["gravity"] = gravity_vector()
    if 'sample_position' not in da.coords:
        da.coords['sample_position'] = sc.vector([0, 0, 0], unit='m')
    da.bins.constituents['data'].variances = da.bins.constituents['data'].values
    for name in ('monitor_1', 'monitor_2'):
        monitor = da.attrs[name].value
        if 'source_position' not in monitor.coords:
            monitor.coords["source_position"] = da.coords['source_position']
        monitor.values[0].variances = monitor.values[0].values
    pixel_shape = da.coords['pixel_shape'].values[0]
    da.coords['pixel_width'] = sc.norm(
        pixel_shape['face1_edge'] - pixel_shape['face1_center']
    ).data
    da.coords['pixel_height'] = sc.norm(
        pixel_shape['face2_center'] - pixel_shape['face1_center']
    ).data
    return da


def load_sample_loki_run(filename: Filename[SampleRun]) -> UnmergedSampleRawData:
    return UnmergedSampleRawData(load_loki_run(filename))


def load_background_loki_run(
    filename: Filename[BackgroundRun],
) -> RawData[BackgroundRun]:
    return RawData[BackgroundRun](load_loki_run(filename))


def load_emptybeam_loki_run(
    filename: Filename[EmptyBeamRun],
) -> RawData[EmptyBeamRun]:
    return RawData[EmptyBeamRun](load_loki_run(filename))


def load_sampletransmission_loki_run(
    filename: Filename[TransmissionRun[SampleRun]],
) -> RawData[TransmissionRun[SampleRun]]:
    return RawData[TransmissionRun[SampleRun]](load_loki_run(filename))


def _merge_run_events(a, b):
    out = a.squeeze().bins.concatenate(b.squeeze())
    for key in a.attrs:
        if key.startswith('monitor'):
            out.attrs[key] = sc.scalar(
                a.attrs[key].value.bins.concatenate(b.attrs[key].value)
            )
    return out


def merge_sample_runs(
    runs: sciline.Series[SampleRunID, UnmergedSampleRawData]
) -> RawData[SampleRun]:
    out = reduce(_merge_run_events, runs.values())
    return RawData[SampleRun](out.bin(tof=1))


def get_monitor(
    da: RawData[RunType], nexus_name: NeXusMonitorName[MonitorType]
) -> RawMonitor[RunType, MonitorType]:
    # See https://github.com/scipp/sciline/issues/52 why copy needed
    return RawMonitor[RunType, MonitorType](da.attrs[nexus_name].value.copy())


def to_logical_dims(da: RawData[RunType]) -> DataWithLogicalDims[RunType]:
    return DataWithLogicalDims[RunType](
        da.fold(
            dim='detector_id', sizes=dict(layer=4, tube=32, straw=7, pixel=512)
        ).flatten(dims=['tube', 'straw'], to='straw')
    )


def detector_straw_mask(
    sample_straws: CalibratedMaskedData[SampleRun],
) -> DetectorLowCountsStrawMask:
    return DetectorLowCountsStrawMask(
        sample_straws.sum(['tof', 'pixel']).data < sc.scalar(300.0, unit='counts')
    )


def detector_beam_stop_mask(
    sample_straws: CalibratedMaskedData[SampleRun],
) -> DetectorBeamStopMask:
    pos = sample_straws.coords['position'].copy()
    pos.fields.z *= 0.0
    return DetectorBeamStopMask((sc.norm(pos) < sc.scalar(0.042, unit='m')))


def detector_tube_edge_mask(
    sample_straws: CalibratedMaskedData[SampleRun],
) -> DetectorTubeEdgeMask:
    return DetectorTubeEdgeMask(
        (abs(sample_straws.coords['position'].fields.x) > sc.scalar(0.36, unit='m'))
        | (abs(sample_straws.coords['position'].fields.y) > sc.scalar(0.28, unit='m'))
    )


def mask_detectors(
    da: DataWithLogicalDims[RunType],
) -> MaskedData[RunType]:
    """Apply pixel-specific masks to raw data.

    Parameters
    ----------
    da:
        Raw data.
    """
    # Beam stop
    da = da.copy(deep=False)
    counts = da.sum('tof').data
    r = sc.sqrt(
        da.coords['position'].fields.x ** 2 + da.coords['position'].fields.y ** 2
    )
    da.masks['low_counts_middle'] = (counts < sc.scalar(20.0, unit='counts')) & (
        r < sc.scalar(0.075, unit='m')
    )
    # Low counts
    da.masks['very_low_counts'] = counts < sc.scalar(3.0, unit='counts')
    return MaskedData[RunType](da)


def mask_after_calibration(
    da: CalibratedMaskedData[RunType],
    lowcounts_straw_mask: Optional[DetectorLowCountsStrawMask],
    beam_stop_mask: Optional[DetectorBeamStopMask],
    tube_edge_mask: Optional[DetectorTubeEdgeMask],
) -> CleanMasked[RunType, Numerator]:
    """Apply pixel-specific masks to raw data.

    Parameters
    ----------
    da:
        Raw data.
    lowcounts_straw_mask:
        Mask for straws with low counts.
    beam_stop_mask:
        Mask for beam stop.
    tube_edge_mask:
        Mask for tube edges.
    """
    da = da.copy(deep=False)
    # Clear masks from beam center finding step, as they are potentially using harsh
    # thresholds which could remove some of the interesting signal.
    da.masks.clear()
    if lowcounts_straw_mask is not None:
        da.masks['low_counts'] = lowcounts_straw_mask
    if beam_stop_mask is not None:
        da.masks['beam_stop'] = beam_stop_mask
    if tube_edge_mask is not None:
        da.masks['tube_edges'] = tube_edge_mask
    return CleanMasked[RunType, Numerator](da)


providers = (
    to_logical_dims,
    detector_straw_mask,
    detector_beam_stop_mask,
    detector_tube_edge_mask,
    get_monitor,
    mask_detectors,
    mask_after_calibration,
    load_background_loki_run,
    load_emptybeam_loki_run,
    load_sample_loki_run,
    load_sampletransmission_loki_run,
    merge_sample_runs,
)
"""
Providers for LoKI
"""
