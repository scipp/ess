# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scippneutron.conversion.tof import wavelength_from_tof
from scippnexus import NXsample, NXsource

from ess.reduce.nexus.types import DetectorBankSizes, Position

from ..reflectometry.conversions import reflectometry_q
from ..reflectometry.types import (
    CoordTransformationGraph,
    DetectorRotation,
    RunType,
    SampleRotation,
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


def coordinate_transformation_graph(
    source_position: Position[NXsource, RunType],
    sample_position: Position[NXsample, RunType],
    sample_rotation: SampleRotation[RunType],
    detector_rotation: DetectorRotation[RunType],
    detector_bank_sizes: DetectorBankSizes,
) -> CoordTransformationGraph[RunType]:
    bank = detector_bank_sizes['multiblade_detector']
    return {
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
        'sample_size': lambda: sc.scalar(20.0, unit='mm'),
        'blade': lambda: sc.arange('blade', bank['blade'] - 1, -1, -1),
        'wire': lambda: sc.arange('wire', bank['wire'] - 1, -1, -1),
        'strip': lambda: sc.arange('strip', bank['strip'] - 1, -1, -1),
        'z_index': lambda blade, wire: blade * wire,
        "wavelength": wavelength_from_tof,
        'sample_rotation': lambda: sample_rotation,
        'detector_rotation': lambda: detector_rotation,
        'source_position': lambda: source_position,
        'sample_position': lambda: sample_position,
    }


def add_coords(
    da: sc.DataArray,
    graph: dict,
) -> sc.DataArray:
    "Adds scattering coordinates to the raw detector data."
    return da.transform_coords(
        (
            "wavelength",
            "theta",
            "divergence_angle",
            "Q",
            "L1",
            "L2",
            "blade",
            "wire",
            "strip",
            "z_index",
            "sample_rotation",
            "detector_rotation",
            "sample_size",
        ),
        graph,
        rename_dims=False,
        keep_intermediate=False,
        keep_aliases=False,
    )


providers = (coordinate_transformation_graph,)
