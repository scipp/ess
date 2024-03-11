# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Loading and merging of LoKI data.
"""
from collections.abc import Iterable, Mapping
from itertools import groupby
from functools import reduce
from typing import Union

import sciline
import scipp as sc

# from ess.reduce import nexus

# from ..sans.common import gravity_vector
from ..sans.types import (
    BackgroundRun,
    # DataFolder,
    DetectorPixelShape,
    Filename,
    # FilenameType,
    # FilePath,
    Incident,
    LabFrameTransform,
    # LoadedDetector,
    # LoadedMonitor,
    LoadedSingleFileDetector,
    LoadedSingleFileMonitor,
    MonitorType,
    NeXusDetectorName,
    NeXusMonitorName,
    PixelShapePath,
    RawData,
    RawMonitor,
    RawSample,
    RawSource,
    RunType,
    SamplePosition,
    SampleRun,
    ScatteringRunType,
    SingleFileDetectorData,
    SourcePosition,
    TransformationPath,
    Transmission,
)


def _merge_events(a, b):
    # Note: the concatenate operation will check that all coordinates are the same.
    return a.bins.concatenate(b)


def _merge_runs(
    data_groups: Mapping[Filename[ScatteringRunType], sc.DataGroup],
    name: Union[
        NeXusDetectorName, NeXusMonitorName[Incident], NeXusMonitorName[Transmission]
    ],
) -> sc.DataGroup:
    """
    Merge events from multiple runs into a single run.
    """
    # TODO: we need some additional checks that the data is compatible. For example,
    # the sample and the source positions should be the same for all runs. Also, the
    # detector geometry (pixel_shapes, lab transform) should be the same for all runs.
    out = next(iter(data_groups.values())).copy(deep=False)
    data_arrays = []
    for dg in data_groups.values():
        events = dg[f'{name}_events']
        if 'event_time_zero' in events.dims:
            events = events.bins.concat('event_time_zero')
        data_arrays.append(events)
    out[f'{name}_events'] = reduce(_merge_events, data_arrays)
    return out


def merge_sample_runs(
    detectors: sciline.Series[Filename[SampleRun], SingleFileDetectorData[SampleRun]],
    detector_name: NeXusDetectorName,
) -> RawData[SampleRun]:
    """
    Merge detector events from multiple sample runs into a single sample run.
    """
    return RawData[SampleRun](_merge_runs(data_groups=detectors, name=detector_name))


def merge_background_runs(
    detectors: sciline.Series[
        Filename[BackgroundRun], LoadedSingleFileDetector[BackgroundRun]
    ],
    detector_name: NeXusDetectorName,
) -> RawData[BackgroundRun]:
    """
    Merge detector events from multiple background runs into a single background run.
    """
    return RawData[BackgroundRun](
        _merge_runs(data_groups=detectors, name=detector_name)
    )


def merge_sample_monitor_runs(
    monitors: sciline.Series[
        Filename[SampleRun], LoadedSingleFileMonitor[SampleRun, MonitorType]
    ],
    monitor_name: NeXusMonitorName[MonitorType],
) -> RawMonitor[SampleRun, MonitorType]:
    """
    Merge monitor events from multiple sample runs into a single sample run.
    """
    return RawMonitor[SampleRun, MonitorType](
        _merge_runs(data_groups=monitors, name=monitor_name)
    )


def merge_background_monitor_runs(
    monitors: sciline.Series[
        Filename[BackgroundRun], LoadedSingleFileMonitor[BackgroundRun, MonitorType]
    ],
    monitor_name: NeXusMonitorName[MonitorType],
) -> RawMonitor[BackgroundRun, MonitorType]:
    """
    Merge monitor events from multiple background runs into a single sample run.
    """
    return RawMonitor[BackgroundRun, MonitorType](
        _merge_runs(data_groups=monitors, name=monitor_name)
    )


def _all_equal(iterable: Iterable) -> bool:
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def merge_sample_positions(
    raw_samples: sciline.Series[Filename[RunType], RawSample[RunType]],
) -> SamplePosition[RunType]:
    sample_positions = [dg['position'] for dg in raw_samples.values()]
    assert _all_equal(
        sample_positions
    ), 'Sample positions must be the same for all runs.'
    return SamplePosition[RunType](sample_positions[0])


def merge_source_positions(
    raw_sources: sciline.Series[Filename[RunType], RawSource[RunType]],
) -> SourcePosition[RunType]:
    source_positions = [dg['position'] for dg in raw_sources.values()]
    assert _all_equal(
        source_positions
    ), 'Source positions must be the same for all runs.'
    return SourcePosition[RunType](source_positions[0])


def merge_detector_pixel_shapes(
    detectors: sciline.Series[
        Filename[ScatteringRunType], LoadedSingleFileDetector[ScatteringRunType]
    ],
    pixel_shape_path: PixelShapePath,
) -> DetectorPixelShape[ScatteringRunType]:
    pixel_shapes = [dg[pixel_shape_path] for dg in detectors.values()]
    assert _all_equal(pixel_shapes), 'Pixel shapes must be the same for all runs.'
    return DetectorPixelShape[ScatteringRunType](pixel_shapes[0])


def merge_lab_frame_transforms(
    detectors: sciline.Series[
        Filename[ScatteringRunType], LoadedSingleFileDetector[ScatteringRunType]
    ],
    transform_path: TransformationPath,
) -> LabFrameTransform[ScatteringRunType]:
    transforms = [dg[transform_path] for dg in detectors.values()]
    assert _all_equal(transforms), 'Lab frame transforms must be the same for all runs.'
    return LabFrameTransform[ScatteringRunType](transforms[0])


providers = (
    merge_sample_runs,
    merge_background_runs,
    merge_sample_monitor_runs,
    merge_background_monitor_runs,
    merge_detector_pixel_shapes,
    merge_lab_frame_transforms,
    merge_sample_positions,
    merge_source_positions,
)
"""Providers to data from multiples files."""
