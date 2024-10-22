# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry.load import load_nx
from ..reflectometry.types import (
    DetectorRotation,
    Filename,
    LoadedNeXusDetector,
    NeXusDetectorName,
    RawDetectorData,
    ReducibleDetectorData,
    RunType,
    SampleRotation,
)
from .geometry import Detector, pixel_coordinate_in_lab_frame
from .types import (
    Chopper1Position,
    Chopper2Position,
    ChopperFrequency,
    ChopperPhase,
    RawChopper,
)


def load_detector(
    file_path: Filename[RunType], detector_name: NeXusDetectorName[RunType]
) -> LoadedNeXusDetector[RunType]:
    return next(load_nx(file_path, f"NXentry/NXinstrument/{detector_name}"))


def load_events(
    detector: LoadedNeXusDetector[RunType], detector_rotation: DetectorRotation[RunType]
) -> RawDetectorData[RunType]:
    detector_numbers = sc.arange(
        "event_id",
        start=1,
        stop=(Detector.nBlades * Detector.nWires * Detector.nStripes).value + 1,
        unit=None,
        dtype="int32",
    )
    data = (
        detector['data']
        .bins.constituents["data"]
        .group(detector_numbers)
        .fold(
            "event_id",
            sizes={
                "blade": Detector.nBlades,
                "wire": Detector.nWires,
                "stripe": Detector.nStripes,
            },
        )
    )
    # Recent versions of scippnexus no longer add variances for events by default, so
    # we add them here if they are missing.
    if data.bins.constituents["data"].data.variances is None:
        data.bins.constituents["data"].data.variances = data.bins.constituents[
            "data"
        ].data.values

    pixel_inds = sc.array(dims=data.dims, values=data.coords["event_id"].values - 1)
    position, angle_from_center_of_beam = pixel_coordinate_in_lab_frame(
        pixelID=pixel_inds, nu=detector_rotation
    )
    data.coords["position"] = position.to(unit="m", copy=False)
    data.coords["angle_from_center_of_beam"] = angle_from_center_of_beam
    return RawDetectorData[RunType](data)


def compute_tof(
    data: RawDetectorData[RunType],
    phase: ChopperPhase[RunType],
    frequency: ChopperFrequency[RunType],
) -> ReducibleDetectorData[RunType]:
    data.bins.coords["tof"] = data.bins.coords.pop("event_time_offset").to(
        unit="ns", dtype="float64", copy=False
    )

    tof_unit = data.bins.coords["tof"].bins.unit
    tau = sc.to_unit(1 / (2 * frequency), tof_unit)
    tof_offset = tau * phase / (180.0 * sc.units.deg)

    event_time_offset = data.bins.coords["tof"]

    minimum = -tof_offset
    frame_bound = tau - tof_offset
    maximum = 2 * tau - tof_offset

    offset = sc.where(
        (minimum < event_time_offset) & (event_time_offset < frame_bound),
        tof_offset,
        sc.where(
            (frame_bound < event_time_offset) & (event_time_offset < maximum),
            tof_offset - tau,
            0.0 * tof_unit,
        ),
    )
    data.bins.masks["outside_of_pulse"] = (minimum > event_time_offset) | (
        event_time_offset > maximum
    )
    data.bins.coords["tof"] += offset
    data.bins.coords["tof"] -= (
        data.coords["angle_from_center_of_beam"].to(unit="deg") / (180.0 * sc.units.deg)
    ) * tau
    return ReducibleDetectorData[RunType](data)


def amor_chopper(f: Filename[RunType]) -> RawChopper[RunType]:
    return next(load_nx(f, "NXentry/NXinstrument/NXdisk_chopper"))


def load_amor_chopper_1_position(ch: RawChopper[RunType]) -> Chopper1Position[RunType]:
    # We know the value has unit 'mm'
    return sc.vector([0, 0, ch["distance"] - ch["pair_separation"] / 2], unit="mm").to(
        unit="m"
    )


def load_amor_chopper_2_position(ch: RawChopper[RunType]) -> Chopper2Position[RunType]:
    # We know the value has unit 'mm'
    return sc.vector([0, 0, ch["distance"] + ch["pair_separation"] / 2], unit="mm").to(
        unit="m"
    )


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


def load_amor_sample_rotation(fp: Filename[RunType]) -> SampleRotation[RunType]:
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


providers = (
    load_detector,
    load_events,
    compute_tof,
    load_amor_ch_frequency,
    load_amor_ch_phase,
    load_amor_chopper_1_position,
    load_amor_chopper_2_position,
    load_amor_sample_rotation,
    load_amor_detector_rotation,
    amor_chopper,
)
