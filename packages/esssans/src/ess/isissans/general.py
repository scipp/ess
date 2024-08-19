# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Providers for the ISIS instruments.
"""

import scipp as sc

from ..sans.types import (
    CalibratedDetector,
    CorrectForGravity,
    DetectorData,
    DetectorIDs,
    DetectorPixelShape,
    DimsToKeep,
    Incident,
    LabFrameTransform,
    MonitorType,
    NeXusMonitorName,
    NonBackgroundWavelengthRange,
    RawMonitor,
    RawMonitorData,
    RunNumber,
    RunTitle,
    RunType,
    SampleRun,
    ScatteringRunType,
    TofData,
    TofMonitor,
    Transmission,
    WavelengthBands,
    WavelengthMask,
)
from .io import LoadedFileContents
from .mantidio import Period


def default_parameters() -> dict:
    return {
        CorrectForGravity: False,
        DimsToKeep: (),
        MonitorPositionOffset[Incident]: MonitorPositionOffset(
            sc.vector([0, 0, 0], unit='m')
        ),
        MonitorPositionOffset[Transmission]: MonitorPositionOffset(
            sc.vector([0, 0, 0], unit='m')
        ),
        DetectorPositionOffset: DetectorPositionOffset(sc.vector([0, 0, 0], unit='m')),
        SamplePositionOffset: SamplePositionOffset(sc.vector([0, 0, 0], unit='m')),
        NonBackgroundWavelengthRange: None,
        WavelengthMask: None,
        WavelengthBands: None,
        Period: None,
    }


def get_detector_data(
    dg: LoadedFileContents[RunType],
    sample_offset: SampleOffset,
    detector_bank_offset: DetectorBankOffset,
) -> DetectorData[RunType]:
    """Get detector data and apply user offsets to raw data.

    Parameters
    ----------
    dg:
        Data loaded with Mantid and converted to Scipp.
    sample_offset:
        Sample offset.
    detector_bank_offset:
        Detector bank offset.
    """
    data = dg['data']
    sample_pos = data.coords['sample_position']
    sample_pos = sample_pos + sample_offset.to(unit=sample_pos.unit, copy=False)
    pos = data.coords['position']
    pos = pos + detector_bank_offset.to(unit=pos.unit, copy=False)
    return DetectorData[RunType](
        dg['data'].assign_coords(position=pos, sample_position=sample_pos)
    )


def assemble_detector_data(
    detector: CalibratedDetector[RunType],
) -> DetectorData[RunType]:
    """Dummy assembly of detector data, detector already contains neutron data."""
    return DetectorData[RunType](detector)


def get_monitor_data(
    dg: LoadedFileContents[RunType], nexus_name: NeXusMonitorName[MonitorType]
) -> RawMonitor[RunType, MonitorType]:
    # See https://github.com/scipp/sciline/issues/52 why copy needed
    mon = dg['monitors'][nexus_name]['data'].copy()
    return RawMonitor[RunType, MonitorType](mon)


def data_to_tof(
    da: DetectorData[ScatteringRunType],
) -> TofData[ScatteringRunType]:
    """Dummy conversion of data to time-of-flight data.
    The data already has a time-of-flight coordinate."""
    return TofData[ScatteringRunType](da)


def monitor_to_tof(
    da: RawMonitorData[RunType, MonitorType],
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


def get_detector_ids_from_sample_run(data: TofData[SampleRun]) -> DetectorIDs:
    """Extract detector IDs from sample run.

    This overrides the function in the masking module which gets the detector IDs from
    the detector before loading event data. In this ISIS case files are loaded using
    Mantid which does not load event separately, so we get IDs from the data.
    """
    return DetectorIDs(
        data.coords[
            'detector_number' if 'detector_number' in data.coords else 'detector_id'
        ]
    )


providers = (
    assemble_detector_data,
    get_detector_data,
    get_detector_ids_from_sample_run,
    get_monitor_data,
    data_to_tof,
    monitor_to_tof,
    run_number,
    run_title,
    lab_frame_transform,
    helium3_tube_detector_pixel_shape,
)
