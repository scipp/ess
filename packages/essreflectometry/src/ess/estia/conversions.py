# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scippneutron.conversion.tof import wavelength_from_tof
from scippnexus import NXsample, NXsource

from ess.reduce.nexus.types import DetectorBankSizes, Position

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
    da: RawDetector[RunType], graph: CoordTransformationGraph[RunType]
) -> DetectorLtotal[RunType]:
    return da.transform_coords(
        ['Ltotal'],
        graph=graph,
    ).coords['Ltotal']


def coordinate_transformation_graph(
    source_position: Position[NXsource, RunType],
    sample_position: Position[NXsample, RunType],
    detector_bank_sizes: DetectorBankSizes,
) -> CoordTransformationGraph[RunType]:
    bank = detector_bank_sizes['multiblade_detector']
    return {
        "wavelength": wavelength_from_tof,
        "theta": theta,
        "divergence_angle": divergence_angle,
        "Q": reflectometry_q,
        "L1": lambda source_position, sample_position: sc.norm(
            sample_position - source_position.to(unit=sample_position.unit)
        ),  # + extra correction for guides?
        "L2": lambda position, sample_position: sc.norm(
            position - sample_position.to(unit=position.unit)
        ),
        "Ltotal": lambda L1, L2: L1.to(unit=L2.unit) + L2,
        'sample_rotation': lambda: sc.scalar(1.0, unit='deg'),
        'detector_rotation': lambda: sc.scalar(3.65, unit='deg'),
        'source_position': lambda: source_position,
        'sample_position': lambda: sample_position,
        'blade': lambda: sc.arange('blade', 0, bank['blade']),
        'wire': lambda: sc.arange('wire', 0, bank['wire']),
        'strip': lambda: sc.arange('strip', 0, bank['strip']),
        'z_index': lambda blade, wire: blade * wire,
        'sample_size': lambda: sc.scalar(20.0, unit='mm'),
    }


def mcstas_wavelength_coordinate_transformation_graph(
    run_type: RunType,
) -> CoordTransformationGraph[RunType]:
    return {
        **coordinate_transformation_graph(),
        "wavelength": lambda wavelength_from_mcstas: wavelength_from_mcstas,
    }


providers = (
    coordinate_transformation_graph,
    detector_ltotal_from_raw,
)
