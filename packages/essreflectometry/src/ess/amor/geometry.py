# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import scipp as sc


class Detector:
    "Description of the geometry of the Amor detector"

    # number of active blades in the detector
    nBlades = sc.scalar(14)
    # number of wires per blade
    nWires = sc.scalar(32)
    # number of stripes per blade
    nStripes = sc.scalar(64)
    # angle of incidence of the beam on the blades (def: 5.1)
    angle = sc.scalar(5.1, unit="degree").to(unit="rad")
    # height-distance of neighboring pixels on one blade
    dZ = sc.scalar(4.0, unit="mm") * sc.sin(angle)
    # depth-distance of neighboring pixels on one blade
    dX = sc.scalar(4.0, unit="mm") * sc.cos(angle)
    # distance between detector blades
    bladeZ = sc.scalar(10.455, unit="mm")
    # vertical center of the detector
    zero = 0.5 * nBlades.value * bladeZ
    # distance from focal point to leading blade edge
    distance = sc.scalar(4000, unit="mm")


def pixel_coordinates_in_detector_system() -> tuple[sc.Variable, sc.Variable]:
    """Determines beam travel distance inside the detector
    and the beam divergence angle from the detector number."""
    pixels = sc.DataArray(
        sc.arange(
            'row',
            1,
            (
                Detector.nBlades * Detector.nWires * Detector.nStripes + sc.scalar(1)
            ).values,
            unit=None,
        ).fold(
            'row',
            sizes={
                'blade': Detector.nBlades,
                'wire': Detector.nWires,
                'stripe': Detector.nStripes,
            },
        ),
        coords={
            'blade': sc.arange('blade', sc.scalar(0), Detector.nBlades),
            'wire': sc.arange('wire', sc.scalar(0), Detector.nWires),
            'stripe': sc.arange('stripe', sc.scalar(0), Detector.nStripes),
        },
    )
    # x position in detector
    # TODO: check with Jochen if this is correct, as old code was:
    # detX = bZi * Detector.dX
    pixels.coords['distance_in_detector'] = (
        Detector.nWires - 1 - pixels.coords['wire']
    ) * Detector.dX
    bladeAngle = 2.0 * sc.asin(0.5 * Detector.bladeZ / Detector.distance)
    pixels.coords['pixel_divergence_angle'] = (
        (Detector.nBlades / 2.0 - pixels.coords['blade']) * bladeAngle
        - sc.atan(
            pixels.coords['wire']
            * Detector.dZ
            / (Detector.distance + pixels.coords['wire'] * Detector.dX)
        )
    ).to(unit='rad')
    pixels.coords['z_index'] = (
        Detector.nWires * pixels.coords['blade'] + pixels.coords['wire']
    )
    return pixels
