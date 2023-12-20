# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Default parameters, providers and utility functions for the loki workflow.
"""


from ..types import (
    DataWithLogicalDims,
    DetectorPixelShape,
    Incident,
    LabFrameTransform,
    LoadedFileContents,
    NexusDetectorName,
    NexusInstrumentPath,
    NeXusMonitorName,
    NexusSourceName,
    RawData,
    RunType,
    TransformationChainPath,
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
    TransformationChainPath: 'transformation_chain',
}


def to_logical_dims(da: RawData[RunType]) -> DataWithLogicalDims[RunType]:
    return DataWithLogicalDims[RunType](
        da.fold(
            dim='detector_number', sizes=dict(layer=4, tube=32, straw=7, pixel=512)
        ).flatten(dims=['tube', 'straw'], to='straw')
    )


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
    transform_path: TransformationChainPath,
) -> LabFrameTransform[RunType]:
    return LabFrameTransform[RunType](
        dg[instrument_path][detector_name][transform_path]
    )


providers = (
    detector_pixel_shape,
    detector_lab_frame_transform,
    to_logical_dims,
)
