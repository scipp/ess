# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc
from ess.reduce.nexus.types import DetectorBankSizes, Position
from scippneutron.conversion import graph
from scippnexus import NXsample

from ..reflectometry import conversions as reflectometry_conversions
from ..reflectometry.types import (
    CoordTransformationGraph,
    DetectorRotation,
    NeutronSourcePosition,
    RunType,
    SampleRotation,
)


def reflectometry_q_x(
    wavelength: sc.Variable, theta: sc.Variable, sample_rotation: sc.Variable
) -> sc.Variable:
    """
    Compute momentum transfer in off-specular direction.

    .. math::
        Q_x = \\frac{2 \\pi}{\\lambda} (cos(\\theta_o) - cos(\\theta_i))

    Where :math:`\\theta_o` is the reflection angle (:func:`theta`)
    and :math:`\\theta_i` is the incident angle on the sample surface.

    This calculation uses the central beam's calibrated incidence angle,
    ``sample_rotation``, as the incidence angle for every event.

    Source:
    `Frédéric Ott, "Off-specular data representations in neutron reflectivity" <https://doi.org/10.1107/S0021889811002858>`_

    Parameters
    ----------
    wavelength:
        Wavelength values for the events.
    theta:
        Angle of reflection for the events.
    sample_rotation:
        Calibrated incidence angle of the central beam.

    Returns
    -------
    :
        Qx-values.
    """
    return reflectometry_conversions.reflectometry_q_x(
        wavelength=wavelength,
        incident_angle=sample_rotation,
        reflection_angle=theta,
    )


def theta(
    divergence_angle: sc.Variable,
    sample_rotation: sc.Variable,
):
    '''Compute the reflection angle in the laboratory x-z plane.

    The central reflected beam must be aligned with the detector reference
    direction. After alignment, the detector rotates by twice each change in
    sample rotation, with the other alignment rotations held fixed.

    Parameters
    ----------
    divergence_angle:
        Signed angle of the outgoing ray from the detector reference direction.
    sample_rotation:
        Calibrated angle between the central reflected beam and the sample surface.

    Returns
    -------
    :
        Angle between the outgoing ray and the sample surface.
    '''
    return divergence_angle + sample_rotation.to(
        unit=divergence_angle.unit, dtype='float64'
    )


def divergence_angle(
    position: sc.Variable,
    sample_position: sc.Variable,
    detector_rotation: sc.Variable,
):
    """Compute the outgoing ray's signed angle from the detector reference direction.

    Use the laboratory x-z plane, with positive angles turning from +z towards +x.
    For an aligned detector, zero corresponds to the central reflected beam.

    Parameters
    ----------
    position:
        Detector pixel position in laboratory coordinates.
    sample_position:
        Sample position in laboratory coordinates.
    detector_rotation:
        Detector reference angle measured from laboratory +z towards +x.

    Returns
    -------
    :
        Signed angular displacement from the detector reference direction.
    """
    p = position - sample_position.to(unit=position.unit)
    return sc.atan2(y=p.fields.x, x=p.fields.z) - detector_rotation.to(
        unit='rad', dtype='float64'
    )


def coordinate_transformation_graph(
    source_position: NeutronSourcePosition[RunType],
    sample_position: Position[NXsample, RunType],
    sample_rotation: SampleRotation[RunType],
    detector_rotation: DetectorRotation[RunType],
    detector_bank_sizes: DetectorBankSizes,
) -> CoordTransformationGraph[RunType]:
    bank = detector_bank_sizes['multiblade_detector']
    return {
        **graph.beamline.beamline(scatter=True),
        "theta": theta,
        "divergence_angle": divergence_angle,
        "Q": reflectometry_conversions.reflectometry_q,
        "Qx": reflectometry_q_x,
        'sample_size': lambda: sc.scalar(20.0, unit='mm'),
        'blade': lambda: sc.arange('blade', bank['blade'] - 1, -1, -1),
        'wire': lambda: sc.arange('wire', bank['wire'] - 1, -1, -1),
        'strip': lambda: sc.arange('strip', bank['strip'] - 1, -1, -1),
        'z_index': lambda blade, wire: blade * wire,
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
            "Qx",
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
