# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Providers for the ISIS instruments.
"""

from dataclasses import dataclass
from typing import Generic, NewType

import sciline
import scipp as sc
import scippnexus as snx

from ess.reduce.nexus.types import NeXusTransformation, Position
from ess.sans.types import (
    BeamCenter,
    DetectorIDs,
    DetectorPixelShape,
    DetectorPositionOffset,
    EmptyDetector,
    EmptyMonitor,
    Incident,
    Measurement,
    MonitorPositionOffset,
    MonitorType,
    NeXusComponent,
    NeXusMonitorName,
    NonBackgroundWavelengthRange,
    RawDetector,
    RawMonitor,
    RunType,
    SampleRun,
    RunType,
    TofDetector,
    TofMonitor,
    Transmission,
)

from .io import LoadedFileContents
from .mantidio import Period


class MonitorOffset(sciline.Scope[MonitorType, sc.Variable], sc.Variable):
    """
    Offset for monitor position for all runs.
    """


@dataclass
class MonitorSpectrumNumber(Generic[MonitorType]):
    """
    Spectrum number for monitor data.

    This is used to identify the monitor in ISIS histogram data files.
    """

    value: int


DetectorBankOffset = NewType('DetectorBankOffset', sc.Variable)
SampleOffset = NewType('SampleOffset', sc.Variable)


def default_parameters() -> dict:
    return {
        MonitorOffset[Incident]: MonitorOffset[Incident](
            sc.vector([0, 0, 0], unit='m')
        ),
        MonitorOffset[Transmission]: MonitorOffset[Transmission](
            sc.vector([0, 0, 0], unit='m')
        ),
        DetectorBankOffset: DetectorBankOffset(sc.vector([0, 0, 0], unit='m')),
        SampleOffset: SampleOffset(sc.vector([0, 0, 0], unit='m')),
        NonBackgroundWavelengthRange: None,
        Period: None,
        # Not used for event files, setting default so the workflow works. If histogram
        # data is used, the user should set this to the correct value.
        MonitorSpectrumNumber[Incident]: MonitorSpectrumNumber[Incident](-1),
        MonitorSpectrumNumber[Transmission]: MonitorSpectrumNumber[Transmission](-1),
    }


def to_detector_position_offset(
    global_offset: DetectorBankOffset, beam_center: BeamCenter
) -> DetectorPositionOffset[RunType]:
    return DetectorPositionOffset[RunType](global_offset - beam_center)


def to_monitor_position_offset(
    global_offset: MonitorOffset[MonitorType],
) -> MonitorPositionOffset[RunType, MonitorType]:
    return MonitorPositionOffset[RunType, MonitorType](global_offset)


def get_source_position(
    dg: LoadedFileContents[RunType],
) -> Position[snx.NXsource, RunType]:
    """Get source position from raw data."""
    return Position[snx.NXsource, RunType](dg['data'].coords['source_position'])


def get_sample_position(
    dg: LoadedFileContents[RunType], offset: SampleOffset
) -> Position[snx.NXsample, RunType]:
    """Get sample position from raw data and apply user offset."""
    return Position[snx.NXsample, RunType](
        dg['data'].coords['sample_position'] + offset.to(unit='m')
    )


def get_detector_data(
    dg: LoadedFileContents[RunType],
) -> NeXusComponent[snx.NXdetector, RunType]:
    """Get detector data and apply user offsets to raw data.

    Parameters
    ----------
    dg:
        Data loaded with Mantid and converted to Scipp.
    """
    # The generic NeXus workflow will try to extract 'data' from this, which is exactly
    # what we also have in the Mantid data. We use the generic workflow since it also
    # applies offsets, etc.
    return NeXusComponent[snx.NXdetector, RunType](dg)


def get_calibrated_isis_detector(
    detector: NeXusComponent[snx.NXdetector, RunType],
    *,
    offset: DetectorPositionOffset[RunType],
) -> EmptyDetector[RunType]:
    """
    Replacement for :py:func:`ess.reduce.nexus.workflow.get_calibrated_detector`.

    Differences:

    - The detector position is already pre-computed.
    - The detector is not reshaped.

    The reason for the partial duplication is to avoid having to put ISIS/Mantid
    specific code in the generic workflow.
    """
    da = detector['data']
    position = detector['data'].coords['position']
    return EmptyDetector[RunType](
        da.assign_coords(position=position + offset.to(unit=position.unit))
    )


def get_monitor_data(
    dg: LoadedFileContents[RunType],
    nexus_name: NeXusMonitorName[MonitorType],
    spectrum_number: MonitorSpectrumNumber[MonitorType],
) -> NeXusComponent[MonitorType, RunType]:
    """
    Extract monitor data that was loaded together with detector data.

    If the raw file is histogram data, Mantid stores this as a Workspace2D, where some
    or all spectra correspond to monitors.

    Parameters
    ----------
    dg:
        Data loaded with Mantid and converted to Scipp.
    nexus_name:
        Name of the monitor in the NeXus file, e.g. 'incident_monitor' or
        'transmission_monitor'. Used when raw data is an EventWorkspace, where
        monitors are stored in a group with this name.
    spectrum_number:
        Spectrum number of the monitor in the NeXus file, e.g., 3 for incident monitor
        or 4 for transmission monitor. This is used when the raw data is a
        Workspace2D, where the monitor data is stored in a spectrum with this number.

    Returns
    -------
    :
        Monitor data extracted from the loaded file.
    """
    # The generic NeXus workflow will try to extract 'data' from this, which is exactly
    # what we also have in the Mantid data. We use the generic workflow since it also
    # applies offsets, etc.
    if 'monitors' in dg:
        # From EventWorkspace
        monitor = dg['monitors'][nexus_name]['data']
    else:
        # From Workspace2D
        monitor = sc.values(dg["data"]["spectrum", sc.index(spectrum_number.value)])
    return NeXusComponent[MonitorType, RunType](
        sc.DataGroup(data=monitor, position=monitor.coords['position'])
    )


def dummy_assemble_detector_data(
    detector: EmptyDetector[RunType],
) -> RawDetector[RunType]:
    """Dummy assembly of detector data, detector already contains neutron data."""
    return RawDetector[RunType](detector)


def dummy_assemble_monitor_data(
    monitor: EmptyMonitor[RunType, MonitorType],
) -> RawMonitor[RunType, MonitorType]:
    """Dummy assembly of monitor data, monitor already contains neutron data."""
    return RawMonitor[RunType, MonitorType](monitor)


def data_to_tof(
    da: RawDetector[RunType],
) -> TofDetector[RunType]:
    """Dummy conversion of data to time-of-flight data.
    The data already has a time-of-flight coordinate."""
    return TofDetector[RunType](da)


def monitor_to_tof(
    da: RawMonitor[RunType, MonitorType],
) -> TofMonitor[RunType, MonitorType]:
    """Dummy conversion of monitor data to time-of-flight data.
    The monitor data already has a time-of-flight coordinate."""
    return TofMonitor[RunType, MonitorType](da)


def experiment_metadata(dg: LoadedFileContents[RunType]) -> Measurement[RunType]:
    """Get experiment metadata from the raw sample data."""
    return Measurement[RunType](
        title=dg['run_title'].value,
        run_number=dg['run_number'],
    )


def helium3_tube_detector_pixel_shape() -> DetectorPixelShape[RunType]:
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


def lab_frame_transform() -> NeXusTransformation[snx.NXdetector, RunType]:
    # Rotate +y to -x
    return NeXusTransformation[snx.NXdetector, RunType](
        sc.spatial.rotation(value=[0, 0, 1 / 2**0.5, 1 / 2**0.5])
    )


def get_detector_ids_from_sample_run(data: TofDetector[SampleRun]) -> DetectorIDs:
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
    dummy_assemble_detector_data,
    dummy_assemble_monitor_data,
    experiment_metadata,
    to_detector_position_offset,
    to_monitor_position_offset,
    get_source_position,
    get_sample_position,
    get_detector_data,
    get_calibrated_isis_detector,
    get_detector_ids_from_sample_run,
    get_monitor_data,
    data_to_tof,
    monitor_to_tof,
    lab_frame_transform,
    helium3_tube_detector_pixel_shape,
)
