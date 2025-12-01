# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scippneutron.conversion.tof import wavelength_from_tof

from ..reflectometry.conversions import reflectometry_q
from ..reflectometry.types import (
    CoordTransformationGraph,
    DetectorLtotal,
    RawDetector,
    RunType,
)


def theta(
    divergence_angle: sc.Variable,
    sample_rotation: sc.Variable,
):
    '''
    Angle of reflection.

    Computes the angle between the scattering direction of
    the neutron and the sample surface.

    Parameters
    ------------
        divergence_angle:
            Divergence angle of the scattered beam.
        sample_rotation:
            Rotation of the sample from to its zero position.

    Returns
    -----------
    The reflection angle of the neutron.
    '''
    return divergence_angle + sample_rotation.to(unit=divergence_angle.unit)


def divergence_angle(
    position: sc.Variable,
    sample_position: sc.Variable,
    detector_rotation: sc.Variable,
):
    """
    Angle between the scattering ray and
    the ray that travels parallel to the sample surface
    when the sample rotation is zero.

    Parameters
    ------------
        position:
            Detector position where the neutron was detected.
        sample_position:
            Position of the sample.
        detector_rotation:
            Rotation of the detector from its zero position.
    Returns
    ----------
    The divergence angle of the scattered beam.
    """
    p = position - sample_position.to(unit=position.unit)
    return sc.atan2(y=p.fields.x, x=p.fields.z) - detector_rotation.to(unit='rad')


def detector_ltotal_from_raw(
    da: RawDetector[RunType], graph: CoordTransformationGraph
) -> DetectorLtotal[RunType]:
    return da.transform_coords(['Ltotal'], graph=graph).coords['Ltotal']


def coordinate_transformation_graph() -> CoordTransformationGraph:
    return {
        "wavelength": wavelength_from_tof,
        "theta": theta,
        "divergence_angle": divergence_angle,
        "Q": reflectometry_q,
        "L1": lambda source_position, sample_position: sc.norm(
            sample_position - source_position
        ),  # + extra correction for guides?
        "L2": lambda position, sample_position: sc.norm(position - sample_position),
        "Ltotal": lambda L1, L2: L1 + L2,
    }


def mcstas_wavelength_coordinate_transformation_graph() -> CoordTransformationGraph:
    return {
        **coordinate_transformation_graph(),
        "wavelength": lambda wavelength_from_mcstas: wavelength_from_mcstas,
    }


providers = (
    coordinate_transformation_graph,
    detector_ltotal_from_raw,
)
