# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc
import scippnexus as snx

from ess.reduce.nexus import workflow as nexus_workflow
from ess.reduce.nexus.types import NeXusFileSpec

from ..reflectometry.load import load_nx
from ..reflectometry.types import (
    Beamline,
    BeamSize,
    DetectorData,
    DetectorRotation,
    Filename,
    Measurement,
    NeXusComponent,
    NeXusDetectorName,
    ProtonCurrent,
    RawSampleRotation,
    RunType,
    SampleRotation,
    SampleRun,
    SampleSize,
)
from .geometry import pixel_coordinates_in_detector_system
from .types import (
    ChopperDistance,
    ChopperFrequency,
    ChopperPhase,
    ChopperSeparation,
    RawChopper,
)


def load_detector(
    file_path: Filename[RunType], detector_name: NeXusDetectorName
) -> NeXusComponent[snx.NXdetector, RunType]:
    return next(load_nx(file_path, f"NXentry/NXinstrument/{detector_name}"))


def load_events(
    detector: NeXusComponent[snx.NXdetector, RunType],
    detector_rotation: DetectorRotation[RunType],
    sample_rotation: SampleRotation[RunType],
    chopper_phase: ChopperPhase[RunType],
    chopper_frequency: ChopperFrequency[RunType],
    chopper_distance: ChopperDistance[RunType],
    chopper_separation: ChopperSeparation[RunType],
    sample_size: SampleSize[RunType],
    beam_size: BeamSize[RunType],
) -> DetectorData[RunType]:
    event_data = detector["data"]
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

    data.coords["sample_rotation"] = sample_rotation.to(unit='rad')
    data.coords["detector_rotation"] = detector_rotation.to(unit='rad')
    data.coords["chopper_phase"] = chopper_phase
    data.coords["chopper_frequency"] = chopper_frequency
    data.coords["chopper_separation"] = chopper_separation
    data.coords["chopper_distance"] = chopper_distance
    data.coords["sample_size"] = sample_size
    data.coords["beam_size"] = beam_size
    return DetectorData[RunType](data)


def amor_chopper(f: Filename[RunType]) -> RawChopper[RunType]:
    return next(load_nx(f, "NXentry/NXinstrument/NXdisk_chopper"))


def load_amor_chopper_distance(ch: RawChopper[RunType]) -> ChopperDistance[RunType]:
    # We know the value has unit 'mm'
    return sc.scalar(ch["distance"], unit="mm")


def load_amor_chopper_separation(ch: RawChopper[RunType]) -> ChopperSeparation[RunType]:
    # We know the value has unit 'mm'
    return sc.scalar(ch["pair_separation"], unit="mm")


def load_amor_ch_phase(ch: RawChopper[RunType]) -> ChopperPhase[RunType]:
    p = ch["phase"]["value"].coords["average_value"].value
    if getattr(p, "unit", None):
        return p
    raise ValueError("No unit was found for the chopper phase")


def load_amor_ch_frequency(ch: RawChopper[RunType]) -> ChopperFrequency[RunType]:
    f = ch["rotation_speed"]["value"].coords["average_value"]
    if getattr(f, "unit", None):
        return f
    raise ValueError("No unit was found for the chopper frequency")


def load_amor_sample_rotation(fp: Filename[RunType]) -> RawSampleRotation[RunType]:
    (mu,) = load_nx(fp, "NXentry/NXinstrument/master_parameters/mu")
    # Jochens Amor code reads the first value of this log
    # see https://github.com/jochenstahn/amor/blob/140e3192ddb7e7f28acee87e2acaee65ce1332aa/libeos/file_reader.py#L272  # noqa: E501
    # might have to change if this field ever becomes truly time-dependent
    return sc.scalar(mu['value'].data['dim_1', 0]['time', 0].value, unit='deg')


def load_amor_detector_rotation(fp: Filename[RunType]) -> DetectorRotation[RunType]:
    (nu,) = load_nx(fp, "NXentry/NXinstrument/master_parameters/nu")
    # Jochens Amor code reads the first value of this log
    # see https://github.com/jochenstahn/amor/blob/140e3192ddb7e7f28acee87e2acaee65ce1332aa/libeos/file_reader.py#L272  # noqa: E501
    # might have to change if this field ever becomes truly time-dependent
    return sc.scalar(nu['value'].data['dim_1', 0]['time', 0].value, unit='deg')


def load_amor_proton_current(
    fp: Filename[RunType],
) -> ProtonCurrent[RunType]:
    (pc,) = load_nx(fp, 'NXentry/NXinstrument/NXdetector/proton_current')
    pc = pc['value']['dim_1', 0]
    pc.data.unit = 'mA/s'
    return pc


def load_beamline_metadata(filename: Filename[SampleRun]) -> Beamline:
    return nexus_workflow.load_beamline_metadata_from_nexus(
        NeXusFileSpec[SampleRun](filename)
    )


def load_measurement_metadata(filename: Filename[SampleRun]) -> Measurement:
    return nexus_workflow.load_measurement_metadata_from_nexus(
        NeXusFileSpec[SampleRun](filename)
    )


providers = (
    load_detector,
    load_events,
    load_amor_ch_frequency,
    load_amor_ch_phase,
    load_amor_chopper_distance,
    load_amor_chopper_separation,
    load_amor_sample_rotation,
    load_amor_detector_rotation,
    load_amor_proton_current,
    amor_chopper,
    load_beamline_metadata,
    load_measurement_metadata,
)
