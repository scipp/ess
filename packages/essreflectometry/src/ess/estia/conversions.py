# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry.conversions import reflectometry_q
from ..reflectometry.types import (
    CoordTransformationGraph,
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


def time_of_flight(
    event_time_offset,
    chopper_open_time,
    chopper_distance,
    Ltotal,
):
    """
    Converts event_time_offset to time_of_flight.
    """
    center_of_pulse = sc.scalar(1.5, unit='ms').to(unit='ns')
    end_of_pulse = sc.scalar(3.0, unit='ms').to(unit='ns')
    cut_time = (
        Ltotal / chopper_distance * (chopper_open_time.to(unit='ns') - end_of_pulse)
        + end_of_pulse
    ).to(unit='ns')

    return (
        sc.where(
            event_time_offset <= cut_time,
            event_time_offset + sc.scalar(1 / 14, unit='s').to(unit='ns'),
            event_time_offset,
        )
        - center_of_pulse
    )


def wavelength(
    time_of_flight,
    Ltotal,
):
    """
    Converts time_of_flight to wavelength.
    """
    return (((sc.constants.h / sc.constants.m_n) / Ltotal) * time_of_flight).to(
        unit='angstrom', copy=False
    )


def mcstas_wavelength_coordinate_transformation_graph() -> CoordTransformationGraph:
    return {
        "wavelength": lambda wavelength_from_mcstas: wavelength_from_mcstas,
        "theta": theta,
        "divergence_angle": divergence_angle,
        "Q": reflectometry_q,
        "L1": lambda source_position, sample_position: sc.norm(
            sample_position - source_position
        ),  # + extra correction for guides?
        "L2": lambda position, sample_position: sc.norm(position - sample_position),
    }


def coordinate_transformation_graph() -> CoordTransformationGraph:
    return {
        "time_of_flight": time_of_flight,
        "wavelength": wavelength,
        "theta": theta,
        "divergence_angle": divergence_angle,
        "Q": reflectometry_q,
        "L1": lambda source_position, sample_position: sc.norm(
            sample_position - source_position
        ),  # + extra correction for guides?
        "L2": lambda position, sample_position: sc.norm(position - sample_position),
        "Ltotal": lambda L1, L2: L1 + L2,
    }


providers = (coordinate_transformation_graph,)
