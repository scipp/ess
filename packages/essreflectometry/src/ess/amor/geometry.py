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


def _pixel_coordinate_in_detector_system(
    pixelID: sc.Variable,
) -> tuple[sc.Variable, sc.Variable]:
    """Determines beam travel distance inside the detector
    and the beam divergence angle from the detector number."""
    (bladeNr, bPixel) = (
        pixelID // (Detector.nWires * Detector.nStripes),
        pixelID % (Detector.nWires * Detector.nStripes),
    )
    # z index on blade, y index on detector
    bZi = bPixel // Detector.nStripes
    # x position in detector
    # TODO: check with Jochen if this is correct, as old code was:
    # detX = bZi * Detector.dX
    distance_inside_detector = (Detector.nWires - 1 - bZi) * Detector.dX

    bladeAngle = (2.0 * sc.asin(0.5 * Detector.bladeZ / Detector.distance)).to(
        unit="degree"
    )
    beam_divergence_angle = (Detector.nBlades / 2.0 - bladeNr) * bladeAngle - (
        sc.atan(bZi * Detector.dZ / (Detector.distance + bZi * Detector.dX))
    ).to(unit="degree")
    return distance_inside_detector, beam_divergence_angle


def pixel_coordinate_in_lab_frame(
    pixelID: sc.Variable, nu: sc.Variable
) -> tuple[sc.Variable, sc.Variable]:
    """Computes spatial coordinates (lab reference frame), and the beam divergence
    angle for the detector pixel associated with `pixelID`"""
    distance_in_detector, divergence_angle = _pixel_coordinate_in_detector_system(
        pixelID
    )

    angle_to_horizon = (nu + divergence_angle).to(unit="rad")
    distance_to_pixel = distance_in_detector + Detector.distance

    global_Y = distance_to_pixel * sc.sin(angle_to_horizon)
    global_Z = distance_to_pixel * sc.cos(angle_to_horizon)
    # TODO: the values for global_X are right now just an estimate. We should check with
    # the instrument scientist what the actual values are. The X positions are ignored
    # in the coordinate transformation, so this is not critical.
    global_X = sc.zeros_like(global_Z) + sc.linspace(
        "stripe", -0.1, 0.1, global_Z.sizes["stripe"], unit="m"
    ).to(unit=global_Z.unit)
    return sc.spatial.as_vectors(global_X, global_Y, global_Z), divergence_angle
