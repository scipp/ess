# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from ess.reduce import nexus

from ..reflectometry.load import load_nx
from ..reflectometry.types import (
    ChopperCorrectedTofEvents,
    DetectorRotation,
    FilePath,
    NeXusDetectorName,
    RawDetector,
    RawEvents,
    Run,
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
    file_path: FilePath[Run], detector_name: NeXusDetectorName[Run]
) -> RawDetector[Run]:
    return nexus.load_detector(file_path=file_path, detector_name=detector_name)


def load_events(
    detector: RawDetector[Run], detector_rotation: DetectorRotation[Run]
) -> RawEvents[Run]:
    detector_numbers = sc.arange(
        "event_id",
        start=1,
        stop=(Detector.nBlades * Detector.nWires * Detector.nStripes).value + 1,
        unit=None,
        dtype="int32",
    )
    data = (
        nexus.extract_detector_data(detector)
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
    x, y, z, angle_from_center_of_beam = pixel_coordinate_in_lab_frame(
        pixelID=pixel_inds, nu=detector_rotation
    )
    data.coords["position"] = sc.spatial.as_vectors(x, y, z).to(unit="m")
    data.coords["angle_from_center_of_beam"] = angle_from_center_of_beam
    return RawEvents[Run](data)


def compute_tof(
    data: RawEvents[Run], phase: ChopperPhase[Run], frequency: ChopperFrequency[Run]
) -> ChopperCorrectedTofEvents[Run]:
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
        data.coords["angle_from_center_of_beam"] / (180.0 * sc.units.deg)
    ) * tau
    return ChopperCorrectedTofEvents[Run](data)


def amor_chopper(f: FilePath[Run]) -> RawChopper[Run]:
    return next(load_nx(f, "NXentry/NXinstrument/NXdisk_chopper"))


def load_amor_chopper_1_position(ch: RawChopper[Run]) -> Chopper1Position[Run]:
    # We know the value has unit 'mm'
    return sc.vector([0, 0, ch["distance"] - ch["pair_separation"] / 2], unit="mm").to(
        unit="m"
    )


def load_amor_chopper_2_position(ch: RawChopper[Run]) -> Chopper2Position[Run]:
    # We know the value has unit 'mm'
    return sc.vector([0, 0, ch["distance"] + ch["pair_separation"] / 2], unit="mm").to(
        unit="m"
    )


def load_amor_ch_phase(ch: RawChopper[Run]) -> ChopperPhase[Run]:
    p = ch["phase"]["value"].coords["average_value"].value
    if getattr(p, "unit", None):
        return p
    raise ValueError("No unit was found for the chopper phase")


def load_amor_ch_frequency(ch: RawChopper[Run]) -> ChopperFrequency[Run]:
    f = ch["rotation_speed"]["value"].coords["average_value"]
    if getattr(f, "unit", None):
        return f
    raise ValueError("No unit was found for the chopper frequency")


def load_amor_sample_rotation(fp: FilePath[Run]) -> SampleRotation[Run]:
    (mu,) = load_nx(fp, "NXentry/NXinstrument/master_parameters/mu")
    # For some reason this is length 1 and not scalar sometimes
    if mu["value"].coords["average_value"].dims != ():
        mu = mu["value"].coords["average_value"].values[0]
    else:
        mu = mu["value"].coords["average_value"].value
    return sc.scalar(mu, unit="deg")


def load_amor_detector_rotation(fp: FilePath[Run]) -> DetectorRotation[Run]:
    (nu,) = load_nx(fp, "NXentry/NXinstrument/master_parameters/nu")
    # For some reason this is length 1 and not scalar sometimes
    if nu["value"].coords["average_value"].dims != ():
        nu = nu["value"].coords["average_value"].values[0]
    else:
        nu = nu["value"].coords["average_value"].value
    return sc.scalar(nu, unit="deg")


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
