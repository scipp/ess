# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Default parameters, providers and utility functions for the loki workflow.
"""


from ..types import (
    DetectorPixelShape,
    Incident,
    LabFrameTransform,
    LoadedFileContents,
    NexusDetectorName,
    NexusInstrumentPath,
    NeXusMonitorName,
    NexusSourceName,
    RunType,
    TransformationPath,
    Transmission,
)

default_parameters = {
    NexusInstrumentPath: 'instrument',
    NexusDetectorName: 'larmor_detector',
    NeXusMonitorName[Incident]: 'monitor_1',
    NeXusMonitorName[Transmission]: 'monitor_2',
    NexusSourceName: 'source',
    # TODO: sample is not in the files, so by not adding the name here, we use the
    # default value of [0, 0, 0] when loading the sample position.
    TransformationPath: 'esssans_loki_transformation',
}


def detector_pixel_shape(
    dg: LoadedFileContents[RunType],
    instrument_path: NexusInstrumentPath,
    detector_name: NexusDetectorName,
) -> DetectorPixelShape[RunType]:
    return DetectorPixelShape[RunType](
        dg[instrument_path][detector_name]['pixel_shape']
    )


def detector_lab_frame_transform(
    dg: LoadedFileContents[RunType],
    instrument_path: NexusInstrumentPath,
    detector_name: NexusDetectorName,
    transform_path: TransformationPath,
) -> LabFrameTransform[RunType]:
    return LabFrameTransform[RunType](
        dg[instrument_path][detector_name][transform_path]
    )


providers = (
    detector_pixel_shape,
    detector_lab_frame_transform,
)
