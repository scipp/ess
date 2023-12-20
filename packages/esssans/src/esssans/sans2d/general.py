# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Loading and masking specific to the ISIS Sans2d instrument and files stored in Scipp's
HDF5 format.
"""
from typing import NewType, Optional

import scipp as sc

from ..types import (
    DataWithLogicalDims,
    DetectorPixelShape,
    Incident,
    LabFrameTransform,
    LoadedFileContents,
    NeXusMonitorName,
    RawData,
    RunNumber,
    RunTitle,
    RunType,
    SampleRun,
    Transmission,
)

ReshapeToLogicalDims = NewType('ReshapeToLogicalDims', bool)
"""Reshape raw data to logical dimensions if True"""


default_parameters = {
    NeXusMonitorName[Incident]: 'monitor2',
    NeXusMonitorName[Transmission]: 'monitor4',
}


def to_logical_dims(
    da: RawData[RunType], reshape: Optional[ReshapeToLogicalDims]
) -> DataWithLogicalDims[RunType]:
    if reshape is None or not reshape:
        return DataWithLogicalDims[RunType](da)
    return DataWithLogicalDims[RunType](
        da.fold(dim='spectrum', sizes={'y': -1, 'x': 1024})
    )


def run_number(dg: LoadedFileContents[SampleRun]) -> RunNumber:
    """Get the run number from the raw sample data."""
    return RunNumber(int(dg['run_number']))


def run_title(dg: LoadedFileContents[SampleRun]) -> RunTitle:
    """Get the run title from the raw sample data."""
    return RunTitle(dg['run_title'].value)


def sans2d_tube_detector_pixel_shape() -> DetectorPixelShape[RunType]:
    # Pixel radius and length
    # found here:
    # https://github.com/mantidproject/mantid/blob/main/instrument/SANS2D_Definition_Tubes.xml
    R = 0.00405
    L = 0.002033984375
    pixel_shape = sc.DataGroup(
        {
            'vertices': sc.vectors(
                dims=['vertex'],
                values=[
                    # Coordinates in pixel-local coordinate system
                    # Bottom face center
                    [0, 0, 0],
                    # Bottom face edge
                    [R, 0, 0],
                    # Top face center
                    [0, L, 0],
                ],
                unit='m',
            ),
            'nexus_class': 'NXcylindrical_geometry',
        }
    )
    return pixel_shape


def lab_frame_transform() -> LabFrameTransform[RunType]:
    # Rotate +y to -x
    return sc.spatial.rotation(value=[0, 0, 1 / 2**0.5, 1 / 2**0.5])


providers = (
    run_number,
    run_title,
    to_logical_dims,
    lab_frame_transform,
    sans2d_tube_detector_pixel_shape,
)
"""
"""
