# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Providers for the ISIS instruments.
"""
import scipp as sc

from ..sans.types import (
    ConfiguredReducibleDataData,
    ConfiguredReducibleMonitor,
    DetectorPixelShape,
    LabFrameTransform,
    MonitorType,
    NeXusMonitorName,
    RawData,
    RawMonitor,
    RunNumber,
    RunTitle,
    RunType,
    SampleRun,
    ScatteringRunType,
    TofData,
    TofMonitor,
)
from .data import LoadedFileContents


def get_detector_data(
    dg: LoadedFileContents[RunType],
) -> RawData[RunType]:
    return RawData[RunType](dg['data'])


def get_monitor_data(
    dg: LoadedFileContents[RunType], nexus_name: NeXusMonitorName[MonitorType]
) -> RawMonitor[RunType, MonitorType]:
    # See https://github.com/scipp/sciline/issues/52 why copy needed
    mon = dg['monitors'][nexus_name]['data'].copy()
    return RawMonitor[RunType, MonitorType](mon)


def data_to_tof(
    da: ConfiguredReducibleDataData[ScatteringRunType],
) -> TofData[ScatteringRunType]:
    """Dummy conversion of data to time-of-flight data.
    The data already has a time-of-flight coordinate."""
    return TofData[ScatteringRunType](da)


def monitor_to_tof(
    da: ConfiguredReducibleMonitor[RunType, MonitorType]
) -> TofMonitor[RunType, MonitorType]:
    """Dummy conversion of monitor data to time-of-flight data.
    The monitor data already has a time-of-flight coordinate."""
    return TofMonitor[RunType, MonitorType](da)


def run_number(dg: LoadedFileContents[SampleRun]) -> RunNumber:
    """Get the run number from the raw sample data."""
    return RunNumber(int(dg['run_number']))


def run_title(dg: LoadedFileContents[SampleRun]) -> RunTitle:
    """Get the run title from the raw sample data."""
    return RunTitle(dg['run_title'].value)


def helium3_tube_detector_pixel_shape() -> DetectorPixelShape[ScatteringRunType]:
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


def lab_frame_transform() -> LabFrameTransform[ScatteringRunType]:
    # Rotate +y to -x
    return sc.spatial.rotation(value=[0, 0, 1 / 2**0.5, 1 / 2**0.5])


providers = (
    get_detector_data,
    get_monitor_data,
    data_to_tof,
    monitor_to_tof,
    run_number,
    run_title,
    lab_frame_transform,
    helium3_tube_detector_pixel_shape,
)
