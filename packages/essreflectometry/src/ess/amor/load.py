# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from ess.reduce import nexus

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
from .types import ChopperFrequency, ChopperPhase


def load_detector(
    file_path: FilePath[Run], detector_name: NeXusDetectorName[Run]
) -> RawDetector[Run]:
    return nexus.load_detector(file_path=file_path, detector_name=detector_name)


def load_events(detector: RawDetector[Run]) -> RawEvents[Run]:
    # Recent versions of scippnexus no longer add variances for events by default, so
    # we add them here if they are missing.
    data = nexus.extract_detector_data(detector)
    if data.bins.constituents['data'].data.variances is None:
        data.bins.constituents['data'].data.variances = data.bins.constituents[
            'data'
        ].data.values
    return RawEvents[Run](data)


class _Detector:
    nBlades = sc.scalar(14)  # number of active blades in the detector
    nWires = sc.scalar(32)  # number of wires per blade
    nStripes = sc.scalar(64)  # number of stipes per blade
    angle = sc.scalar(5.1, unit='degree').to(
        unit='rad'
    )  # deg  angle of incidence of the beam on the blades (def: 5.1)
    dZ = sc.scalar(4.0, unit='mm') * sc.sin(
        angle
    )  # mm  height-distance of neighboring pixels on one blade
    dX = sc.scalar(4.0, unit='mm') * sc.cos(
        angle
    )  # mm  depth-distance of neighboring pixels on one blace
    bladeZ = sc.scalar(10.455, unit='mm')  # mm  distance between detector blades
    zero = 0.5 * nBlades.value * bladeZ  # mm  vertical center of the detector
    distance = sc.scalar(
        4000, unit='mm'
    )  # mm  distance from focal point to leading blade edge


def _pixel_coordinate_in_detector_system(pixelID):
    """determine spatial coordinates and angles from pixel number"""
    pixelID.unit = ''
    (bladeNr, bPixel) = pixelID // (_Detector.nWires * _Detector.nStripes), pixelID % (
        _Detector.nWires * _Detector.nStripes
    )
    (bZi, detYi) = (
        bPixel // _Detector.nStripes,
        bPixel % _Detector.nStripes,
    )  # z index on blade, y index on detector
    detZi = bladeNr * _Detector.nWires + bZi  # z index on detector
    detX = bZi * _Detector.dX  # x position in detector

    bladeAngle = (2.0 * sc.asin(0.5 * _Detector.bladeZ / _Detector.distance)).to(
        unit='degree'
    )
    delta = (_Detector.nBlades / 2.0 - bladeNr) * bladeAngle - (
        sc.atan(bZi * _Detector.dZ / (_Detector.distance + bZi * _Detector.dX))
    ).to(unit='degree')

    # z is in the direction of the center of the beam, y is the direction 'up'
    pixelID.unit = None
    return detYi, detZi, detX, delta


def _pixel_coordinate_in_lab_frame(pixelID, nu):
    _, detZi, detX, delta = _pixel_coordinate_in_detector_system(pixelID)

    angle_to_horizon = (nu + delta).to(unit='rad')
    distance_to_pixel = detX + _Detector.distance
    # TODO: put the correct value here
    global_X = sc.zeros(dims=pixelID.dims, shape=pixelID.shape, unit='mm')
    global_Y = distance_to_pixel * sc.sin(angle_to_horizon)
    global_Z = distance_to_pixel * sc.cos(angle_to_horizon)
    return global_X, global_Y, global_Z


def add_position(
    detector: RawDetector[Run], mu: SampleRotation[Run], nu: DetectorRotation[Run]
) -> RawEvents[Run]:
    events = load_events(detector)
    if 'event_time_zero' in events.coords:
        events.bins.coords['event_time_zero'] = sc.bins_like(
            events, fill_value=events.coords['event_time_zero']
        )
    events = events.bins.concat('event_time_zero').data.value
    events = events.copy()
    events.coords['detector_number'] = events.coords.pop('event_id')

    pixelID = events.coords['detector_number']
    x, y, z = _pixel_coordinate_in_lab_frame(pixelID, nu)
    position = sc.spatial.as_vectors(x, y, z)
    # TODO: include this or not?
    # position.fields.y += position.fields.z * sc.tan(
    #    mu - (0.955 * sc.units.deg)
    # )
    position = position.to(unit='m')
    events.coords['position'] = position
    return events


def compute_tof(
    data: RawEvents[Run], phase: ChopperPhase[Run], frequency: ChopperFrequency[Run]
) -> ChopperCorrectedTofEvents[Run]:
    dim = 'tof'
    data.coords[dim] = data.coords.pop('event_time_offset').to(
        unit='us', dtype='float64', copy=False
    )
    tof_unit = data.coords[dim].unit
    tau = sc.to_unit(1 / (2 * frequency), tof_unit)
    tof_offset = tau * phase / (180.0 * sc.units.deg)
    # TODO: Add other offset - taking into account flight time from chopper to detector
    # Make 2 bins, one for each pulse
    edges = sc.concat([-tof_offset, tau - tof_offset, 2 * tau - tof_offset], dim)
    data = data.bin({dim: sc.to_unit(edges, tof_unit)})
    # Make one offset for each bin
    offset = sc.concat([tof_offset, tof_offset - tau], dim)
    # TODO: Add other offset here as well
    # Apply the offset on both bins
    data.bins.coords[dim] += offset
    # Rebin to exclude second (empty) pulse range
    # TODO: this or that?
    # data = data.bin({dim: sc.concat([0.0 * sc.units.us, tau], dim)})
    data = data.bin({dim: sc.concat([0.0 * sc.units.us, 2 * tau], dim)})
    data = data.squeeze().values
    return ChopperCorrectedTofEvents[Run](data)


providers = (load_detector, load_events, add_position, compute_tof)
