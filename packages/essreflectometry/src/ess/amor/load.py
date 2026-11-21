# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc
from ess.reduce.nexus import workflow as nexus_workflow
from ess.reduce.nexus.types import NeXusFileSpec

from ..reflectometry.load import load_nx
from ..reflectometry.types import (
    Beamline,
    DetectorRotation,
    Filename,
    Measurement,
    NeXusDetectorName,
    RawChopper,
    RawDetector,
    RawSampleRotation,
    RunType,
    SampleRun,
)
from .geometry import pixel_coordinates_in_detector_system
from .types import (
    ChopperDistance,
    ChopperFrequency,
    ChopperSeparation,
    ProtonCurrent,
    PulseTimes,
)


def group_events_by_pixel(event_data: sc.DataArray) -> sc.DataArray:
    """Reshape events binned by pulse to events binned by detector pixel.

    Parameters
    ----------
    event_data:
        Events as loaded from an ``NXevent_data`` group, binned by pulse.

    Returns
    -------
    :
        The events binned by ``blade``, ``wire`` and ``strip``. The pulse
        time of each event is kept in the event coordinate ``event_time_zero``.
    """
    if 'event_time_zero' in event_data.coords:
        event_data.bins.coords['event_time_zero'] = sc.bins_like(
            event_data, fill_value=event_data.coords['event_time_zero']
        )

    detector_numbers = pixel_coordinates_in_detector_system()
    data = (
        event_data.bins.constituents["data"]
        .group(detector_numbers.data.flatten(to='event_id'))
        .fold("event_id", sizes=detector_numbers.sizes)
    )
    data.coords.update(detector_numbers.coords)

    if data.bins.constituents["data"].data.variances is None:
        data.bins.constituents["data"].data.variances = data.bins.constituents[
            "data"
        ].data.values

    return data


def load_events(
    file_path: Filename[RunType], detector_name: NeXusDetectorName
) -> RawDetector[RunType]:
    """Load the events of the detector and bin them by detector pixel."""
    (events,) = load_nx(file_path, f"NXentry/{detector_name}/data")
    return RawDetector[RunType](group_events_by_pixel(events))


def amor_chopper(f: Filename[RunType]) -> RawChopper[RunType]:
    """Load the chopper group from a NeXus file."""
    return next(load_nx(f, "NXentry/NXinstrument/NXdisk_chopper"))


def load_amor_chopper_distance(ch: RawChopper[RunType]) -> ChopperDistance[RunType]:
    """Extract the chopper distance from the chopper group."""
    # We know the value has unit 'mm'
    return sc.scalar(ch["distance"], unit="mm")


def load_amor_chopper_separation(ch: RawChopper[RunType]) -> ChopperSeparation[RunType]:
    """Extract the chopper pair separation from the chopper group."""
    # We know the value has unit 'mm'
    return sc.scalar(ch["pair_separation"], unit="mm")


def load_amor_chopper_frequency(ch: RawChopper[RunType]) -> ChopperFrequency[RunType]:
    """Extract the rotation frequency of the choppers from the chopper group."""
    # Scipp does not convert between 'rpm' and 'Hz'
    speed = ch["rotation_speed"]
    if speed.unit != 'rpm':
        raise ValueError(f"Expected the chopper speed in 'rpm', got '{speed.unit}'")
    return (speed.value / sc.scalar(1.0, unit='min')).to(unit='Hz')


def load_amor_sample_rotation(fp: Filename[RunType]) -> RawSampleRotation[RunType]:
    """Load the angle between the instrument horizon and the sample surface."""
    (mu,) = load_nx(fp, "NXentry/measurement_configuration/mu")
    return mu


def load_amor_detector_rotation(fp: Filename[RunType]) -> DetectorRotation[RunType]:
    """Load the angle between the instrument horizon and the detector center."""
    (nu,) = load_nx(fp, "NXentry/measurement_configuration/nu")
    return nu


def load_amor_proton_current(
    fp: Filename[RunType], detector_name: NeXusDetectorName
) -> ProtonCurrent[RunType]:
    """Load the log of the proton current from the NeXus detector group."""
    (log,) = load_nx(fp, f"NXentry/{detector_name}/proton_current")
    # A log with alarms is loaded as a group that holds the values and the alarms.
    if isinstance(log, sc.DataGroup):
        log = log["value"]
    return ProtonCurrent[RunType](log)


def load_amor_pulse_times(fp: Filename[RunType]) -> PulseTimes[RunType]:
    """Load the start times of all chopper revolutions from the beam monitor."""
    (times,) = load_nx(fp, "NXentry/beam_monitor/time")
    return PulseTimes[RunType](times.rename_dims({times.dim: 'time'}))


def load_beamline_metadata(filename: Filename[RunType]) -> Beamline[RunType]:
    """Load beamline metadata from a NeXus file."""
    return nexus_workflow.load_beamline_metadata_from_nexus(
        NeXusFileSpec[SampleRun](filename)
    )


def load_measurement_metadata(filename: Filename[RunType]) -> Measurement[RunType]:
    """Load measurement metadata from a NeXus file."""
    return nexus_workflow.load_measurement_metadata_from_nexus(
        NeXusFileSpec[SampleRun](filename)
    )


providers = (
    load_events,
    load_amor_chopper_frequency,
    load_amor_chopper_distance,
    load_amor_chopper_separation,
    load_amor_sample_rotation,
    load_amor_detector_rotation,
    load_amor_proton_current,
    load_amor_pulse_times,
    amor_chopper,
    load_beamline_metadata,
    load_measurement_metadata,
)
