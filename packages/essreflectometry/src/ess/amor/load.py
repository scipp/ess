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
)
from .types import ChopperFrequency, ChopperPhase


class Detector:
    'Description of the geometry of the Amor detector'
    # number of active blades in the detector
    nBlades = sc.scalar(14)
    # number of wires per blade
    nWires = sc.scalar(32)
    # number of stipes per blade
    nStripes = sc.scalar(64)
    # angle of incidence of the beam on the blades (def: 5.1)
    angle = sc.scalar(5.1, unit='degree').to(unit='rad')
    # height-distance of neighboring pixels on one blade
    dZ = sc.scalar(4.0, unit='mm') * sc.sin(angle)
    # depth-distance of neighboring pixels on one blade
    dX = sc.scalar(4.0, unit='mm') * sc.cos(angle)
    # distance between detector blades
    bladeZ = sc.scalar(10.455, unit='mm')
    # vertical center of the detector
    zero = 0.5 * nBlades.value * bladeZ
    # distance from focal point to leading blade edge
    distance = sc.scalar(4000, unit='mm')


def _pixel_coordinate_in_detector_system(pixelID):
    """determine spatial coordinates and angles from pixel number"""
    pixelID.unit = ''
    (bladeNr, bPixel) = pixelID // (Detector.nWires * Detector.nStripes), pixelID % (
        Detector.nWires * Detector.nStripes
    )
    # z index on blade, y index on detector
    (bZi, detYi) = (
        bPixel // Detector.nStripes,
        bPixel % Detector.nStripes,
    )
    # z index on detector
    detZi = bladeNr * Detector.nWires + bZi
    # x position in detector
    detX = bZi * Detector.dX

    bladeAngle = (2.0 * sc.asin(0.5 * Detector.bladeZ / Detector.distance)).to(
        unit='degree'
    )
    delta = (Detector.nBlades / 2.0 - bladeNr) * bladeAngle - (
        sc.atan(bZi * Detector.dZ / (Detector.distance + bZi * Detector.dX))
    ).to(unit='degree')

    # z is in the direction of the center of the beam, y is the direction 'up'
    pixelID.unit = None
    return detYi, detZi, detX, delta


def _pixel_coordinate_in_lab_frame(pixelID, nu):
    _, detZi, detX, delta = _pixel_coordinate_in_detector_system(pixelID)

    angle_to_horizon = (nu + delta).to(unit='rad')
    distance_to_pixel = detX + Detector.distance
    # TODO: put the correct value here
    global_X = sc.zeros(dims=pixelID.dims, shape=pixelID.shape, unit='mm')
    global_Y = distance_to_pixel * sc.sin(angle_to_horizon)
    global_Z = distance_to_pixel * sc.cos(angle_to_horizon)
    return global_X, global_Y, global_Z


def load_detector(
    file_path: FilePath[Run], detector_name: NeXusDetectorName[Run]
) -> RawDetector[Run]:
    return nexus.load_detector(file_path=file_path, detector_name=detector_name)


def load_events(
    detector: RawDetector[Run], detector_rotation: DetectorRotation[Run]
) -> RawEvents[Run]:
    detector_numbers = sc.arange(
        'event_id',
        start=0,
        stop=(Detector.nBlades * Detector.nWires * Detector.nStripes).value,
        unit=None,
        dtype='int32',
    )
    data = (
        nexus.extract_detector_data(detector)
        .bins.constituents['data']
        .group(detector_numbers)
        .fold(
            'event_id',
            sizes={
                'blade': Detector.nBlades,
                'wire': Detector.nWires,
                'stipe': Detector.nStripes,
            },
        )
    )
    # Recent versions of scippnexus no longer add variances for events by default, so
    # we add them here if they are missing.
    if data.bins.constituents['data'].data.variances is None:
        data.bins.constituents['data'].data.variances = data.bins.constituents[
            'data'
        ].data.values

    data.coords['position'] = sc.spatial.as_vectors(
        *_pixel_coordinate_in_lab_frame(
            data.coords['event_id'],
            detector_rotation,
        )
    ).to(unit='m')
    return RawEvents[Run](data)


def compute_tof(
    data: RawEvents[Run], phase: ChopperPhase[Run], frequency: ChopperFrequency[Run]
) -> ChopperCorrectedTofEvents[Run]:
    dim = 'tof'
    data.bins.coords[dim] = data.bins.coords.pop('event_time_offset').to(
        unit='ns', dtype='float64', copy=False
    )

    tof_unit = data.bins.coords[dim].bins.unit
    tau = sc.to_unit(1 / (2 * frequency), tof_unit)
    tof_offset = tau * phase / (180.0 * sc.units.deg)

    event_time_offset = data.bins.coords[dim]

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
    data.bins.masks['outside_of_pulse'] = (minimum > event_time_offset) | (
        event_time_offset > maximum
    )
    data.bins.coords[dim] += offset

    return ChopperCorrectedTofEvents[Run](data)


providers = (load_detector, load_events, compute_tof)
