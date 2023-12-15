# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Loading and masking specific to the ISIS Sans2d instrument and files stored in Scipp's
HDF5 format.
"""
from functools import reduce
from pathlib import Path
from typing import NewType, Optional, Union

import sciline
import scipp as sc
import scippnexus as snx

from ..common import gravity_vector
from ..types import (
    BackgroundRun,
    CalibratedMaskedData,
    CleanMasked,
    DataWithLogicalDims,
    # DetectorDataEntryName,
    DetectorPixelShape,
    EmptyBeamRun,
    Filename,
    Incident,
    LabFrameTransform,
    LoadedDetectorContents,
    LoadedMonitorContents,
    MaskedData,
    # MonitorDataEntryName,
    MonitorType,
    NexusDetectorName,
    NexusInstrumentPath,
    NeXusMonitorName,
    NexusSampleName,
    NexusSourceName,
    Numerator,
    PatchedData,
    PatchedMonitor,
    RawData,
    RawMonitor,
    RunID,
    RunType,
    SamplePosition,
    SampleRun,
    SourcePosition,
    TofData,
    TofMonitor,
    Transmission,
    TransmissionRun,
    UnmergedRawData,
    UnmergedRawMonitor,
)

DetectorLowCountsStrawMask = NewType('DetectorLowCountsStrawMask', sc.Variable)
"""Detector low-counts straw mask"""
DetectorBadStrawsMask = NewType('DetectorBadStrawsMask', sc.Variable)
"""Detector bad straws mask"""
DetectorBeamStopMask = NewType('DetectorBeamStopMask', sc.Variable)
"""Detector beam stop mask"""
DetectorTubeEdgeMask = NewType('DetectorTubeEdgeMask', sc.Variable)
"""Detector tube edge mask"""


default_parameters = {
    NexusInstrumentPath: 'entry/instrument',
    NexusDetectorName: 'larmor_detector',
    NeXusMonitorName[Incident]: 'monitor_1',
    NeXusMonitorName[Transmission]: 'monitor_2',
    NexusSourceName: 'source',
    # TODO: sample is not in the files, so by not adding the name here, we use the
    # default value of [0, 0, 0] when loading the sample position.
}


def _load_file_entry(filename: str, entry: Union[str, Path]) -> sc.DataArray:
    from .data import get_path

    with snx.File(get_path(filename)) as f:
        dg = f[str(entry)][()]
    dg = snx.compute_positions(dg, store_transform='transformation_chain')

    return dg


#     # # TODO: Use the new scippnexus to avoid using load_nexus, now that transformations
#     # # are supported.
#     # da = scn.load_nexus(get_path(filename))
#     # if 'gravity' not in da.coords:
#     #     da.coords["gravity"] = gravity_vector()
#     # if 'sample_position' not in da.coords:
#     #     da.coords['sample_position'] = sc.vector([0, 0, 0], unit='m')
#     # da.bins.constituents['data'].variances = da.bins.constituents['data'].values
#     # for name in ('monitor_1', 'monitor_2'):
#     #     monitor = da.attrs[name].value
#     #     if 'source_position' not in monitor.coords:
#     #         monitor.coords["source_position"] = da.coords['source_position']
#     #     monitor.values[0].variances = monitor.values[0].values
#     # pixel_shape = da.coords['pixel_shape'].values[0]
#     # da.coords['pixel_width'] = sc.norm(
#     #     pixel_shape['face1_edge'] - pixel_shape['face1_center']
#     # ).data
#     # da.coords['pixel_height'] = sc.norm(
#     #     pixel_shape['face2_center'] - pixel_shape['face1_center']
#     # ).data
#     # return da


# def _load_detector_data(filename: str, ent) -> sc.DataArray:
#     return _load_file_entry(filename=filename, entry='entry/instrument/larmor_detector')
#     # return _load_file_entry(filename=filename, entry='entry')


def load_data_run(
    filename: Filename[RunType],
    instrument_path: NexusInstrumentPath,
    detector_name: NexusDetectorName,
) -> LoadedDetectorContents[RunType]:
    entry = Path(instrument_path) / Path(detector_name)
    dg = _load_file_entry(filename=filename, entry=entry)
    # dg = _load_file_entry(filename=filename, entry='entry')
    return LoadedDetectorContents[RunType](dg)


def get_detector_data(
    dg: LoadedDetectorContents[RunType],
    detector_name: NexusDetectorName,
) -> UnmergedRawData[RunType]:
    da = dg[f'{detector_name}_events']
    # out = da.fold(
    #     dim='detector_id', sizes=dict(layer=4, tube=32, straw=7, pixel=512)
    # ).flatten(dims=['tube', 'straw'], to='straw')
    return UnmergedRawData[RunType](da)


# def get_detector_data(
#     dg: LoadedDetectorContents[RunType],
#     entry: DetectorDataEntryName[RunType],
# ) -> UnmergedRawData[RunType]:
#     return UnmergedRawData[RunType](dg[f"{entry.split('/')[-1]}_events"])


# def load_emptybeam_run(
#     filename: Filename[EmptyBeamRun]
# ) -> LoadedDetectorContents[EmptyBeamRun]:
#     return LoadedDetectorContents[EmptyBeamRun](_load_detector_data(filename))


# def load_sample_transmission_run(
#     filename: Filename[TransmissionRun[SampleRun]],
# ) -> RawData[TransmissionRun[SampleRun]]:
#     return RawData[TransmissionRun[SampleRun]](_load_detector_data(filename))


# def load_background_transmission_run(
#     filename: Filename[TransmissionRun[BackgroundRun]],
# ) -> RawData[TransmissionRun[BackgroundRun]]:
#     return RawData[TransmissionRun[BackgroundRun]](_load_detector_data(filename))


def _merge_events(a, b):
    return a.squeeze().bins.concatenate(b.squeeze())
    # for key in a.attrs:
    #     if key.startswith('monitor'):
    #         out.attrs[key] = sc.scalar(
    #             a.attrs[key].value.bins.concatenate(b.attrs[key].value)
    #         )
    # return out


def merge_detector_events(
    runs: sciline.Series[RunID[RunType], UnmergedRawData[RunType]]
) -> RawData[RunType]:
    # return RawData[RunType](list(runs.values())[0])  # .bin(tof=1))
    return RawData[RunType](reduce(_merge_events, runs.values()))  # .bin(tof=1))


def merge_monitor_events(
    runs: sciline.Series[RunID[RunType], UnmergedRawMonitor[RunType, MonitorType]]
) -> RawMonitor[RunType, MonitorType]:
    return RawMonitor[RunType, MonitorType](reduce(_merge_events, runs.values()))


def load_monitor(
    filename: Filename[RunType],
    instrument_path: NexusInstrumentPath,
    monitor_name: NeXusMonitorName[MonitorType],
) -> LoadedMonitorContents[RunType, MonitorType]:
    print(monitor_name)
    entry = Path(instrument_path) / Path(monitor_name)
    dg = _load_file_entry(filename=filename, entry=entry)
    return LoadedMonitorContents[RunType, MonitorType](dg)


def get_monitor_data(
    dg: LoadedMonitorContents[RunType, MonitorType],
    monitor_name: NeXusMonitorName[MonitorType],
) -> UnmergedRawMonitor[RunType, MonitorType]:
    return UnmergedRawMonitor[RunType, MonitorType](dg[f'{monitor_name}_events'])


def load_sample_position(
    filename: Filename[RunType],
    instrument_path: NexusInstrumentPath,
    sample_name: Optional[NexusSampleName],
) -> SamplePosition[RunType]:
    # TODO: sample_name is optional for now because it is not found in all the files.
    if sample_name is None:
        out = sc.vector(value=[0, 0, 0], unit='m')
    else:
        entry = Path(instrument_path) / Path(sample_name)
        dg = _load_file_entry(filename=filename, entry=entry)
        out = SamplePosition[RunType](dg['position'])
    return SamplePosition[RunType](out)


def load_source_position(
    filename: Filename[RunType],
    instrument_path: NexusInstrumentPath,
    source_name: NexusSourceName,
) -> SourcePosition[RunType]:
    entry = Path(instrument_path) / Path(source_name)
    dg = _load_file_entry(filename=filename, entry=entry)
    return SourcePosition[RunType](dg['position'])


def patch_detector_data(
    da: RawData[RunType],
    sample_position: SamplePosition[RunType],
    source_position: SourcePosition[RunType],
) -> PatchedData[RunType]:
    da.coords['sample_position'] = sample_position
    da.coords['source_position'] = source_position
    return PatchedData[RunType](da)


def patch_monitor_data(
    da: RawMonitor[RunType, MonitorType],
    source_position: SourcePosition[RunType],
) -> PatchedMonitor[RunType, MonitorType]:
    da.coords['source_position'] = source_position
    return PatchedMonitor[RunType, MonitorType](da)


def _convert_to_tof(da: sc.DataArray) -> sc.DataArray:
    da.bins.coords['tof'] = da.bins.coords.pop('event_time_offset')
    return da


def convert_detector_to_tof(
    da: RawData[RunType],
) -> TofData[RunType]:
    # TODO: This is where the frame unwrapping would occur
    return TofData[RunType](_convert_to_tof(da))


def convert_monitor_to_tof(
    da: RawMonitor[RunType, MonitorType],
) -> TofMonitor[RunType, MonitorType]:
    return TofMonitor[RunType, MonitorType](_convert_to_tof(da))


# def detector_pixel_shape() -> DetectorPixelShape[RunType]:


def to_logical_dims(da: TofData[RunType]) -> DataWithLogicalDims[RunType]:
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


def detector_pixel_shape(
    data_groups: sciline.Series[RunID[RunType], LoadedDetectorContents[RunType]],
) -> DetectorPixelShape[RunType]:
    return DetectorPixelShape[RunType](list(data_groups.values())[0]['pixel_shape'])


def detector_lab_frame_transform(
    data_groups: sciline.Series[RunID[RunType], LoadedDetectorContents[RunType]],
) -> LabFrameTransform[RunType]:
    return LabFrameTransform[RunType](
        list(data_groups.values())[0]['transformation_chain']
    )


providers = (
    to_logical_dims,
    detector_straw_mask,
    detector_beam_stop_mask,
    detector_tube_edge_mask,
    detector_pixel_shape,
    detector_lab_frame_transform,
    get_detector_data,
    get_monitor_data,
    load_monitor,
    mask_detectors,
    mask_after_calibration,
    load_data_run,
    load_sample_position,
    load_source_position,
    # load_emptybeam_run,
    # load_sample_transmission_run,
    # load_background_transmission_run,
    merge_detector_events,
    merge_monitor_events,
    patch_detector_data,
    patch_monitor_data,
    convert_detector_to_tof,
    convert_monitor_to_tof,
)

# providers = (
#     to_logical_dims,
#     detector_straw_mask,
#     detector_beam_stop_mask,
#     detector_tube_edge_mask,
#     get_monitor,
#     mask_detectors,
#     mask_after_calibration,
#     load_loki_data_run,
#     load_loki_emptybeam_run,
#     load_loki_sample_transmission_run,
#     load_loki_background_transmission_run,
#     merge_runs,
# )
# """
# Providers for LoKI
# """
