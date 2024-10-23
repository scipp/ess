# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.constants import pi
from scippneutron._utils import elem_dtype, elem_unit
from scippneutron.conversion.beamline import scattering_angle_in_yz_plane
from scippneutron.conversion.graph import beamline, tof

from .types import (
    BeamDivergenceLimits,
    DataWithScatteringCoordinates,
    DetectorRotation,
    Gravity,
    IncidentBeam,
    MaskedData,
    ReducibleDetectorData,
    RunType,
    SamplePosition,
    SampleRotation,
    SpecularReflectionCoordTransformGraph,
    WavelengthBins,
    YIndexLimits,
    ZIndexLimits,
)


def theta(
    incident_beam: sc.Variable,
    scattered_beam: sc.Variable,
    wavelength: sc.Variable,
    gravity: sc.Variable,
    sample_rotation: sc.Variable,
) -> sc.Variable:
    r"""Compute the scattering angle w.r.t. the sample plane.

    This function uses the definition given in
    :func:`scippneutron.conversion.beamline.scattering_angle_in_yz_plane`
    and includes the sample rotation :math:`\omega`:

    .. math::

        \mathsf{tan}(\gamma) &= \frac{|y_d^{\prime}|}{z_d} \\
        \theta = \gamma - \omega

    with

    .. math::

        y'_d = y_d + \frac{|g| m_n^2}{2 h^2} L_2^{\prime\, 2} \lambda^2

    Attention
    ---------
        The above equation for :math:`y'_d` approximates :math:`L_2 \approx L'_2`.
        See :func:`scippneutron.conversion.beamline.scattering_angle_in_yz_plane`
        for more information.

    Parameters
    ----------
    incident_beam:
        Beam from source to sample. Expects ``dtype=vector3``.
    scattered_beam:
        Beam from sample to detector. Expects ``dtype=vector3``.
    wavelength:
        Wavelength of neutrons.
    gravity:
        Gravity vector.
    sample_rotation:
        The sample rotation angle :math:`\omega`.

    Returns
    -------
    :
        The polar scattering angle :math:`\theta`.
    """
    angle = scattering_angle_in_yz_plane(
        incident_beam=incident_beam,
        scattered_beam=scattered_beam,
        wavelength=wavelength,
        gravity=gravity,
    )
    angle -= sample_rotation.to(unit=elem_unit(angle))
    return angle


def reflectometry_q(wavelength: sc.Variable, theta: sc.Variable) -> sc.Variable:
    """
    Compute the Q vector from the theta angle computed as the difference
    between gamma and omega.
    Note that this is identical the 'normal' Q defined in scippneutron, except that
    the `theta` angle is given as an input instead of `two_theta`.

    Parameters
    ----------
    wavelength:
        Wavelength values for the events.
    theta:
        Theta values, accounting for gravity.

    Returns
    -------
    :
        Q-values.
    """
    dtype = elem_dtype(wavelength)
    c = (4 * pi).astype(dtype)
    return c * sc.sin(theta.astype(dtype, copy=False)) / wavelength


def specular_reflection(
    incident_beam: IncidentBeam[RunType],
    sample_position: SamplePosition[RunType],
    sample_rotation: SampleRotation[RunType],
    detector_rotation: DetectorRotation[RunType],
    gravity: Gravity,
) -> SpecularReflectionCoordTransformGraph[RunType]:
    """
    Generate a coordinate transformation graph for specular reflection reflectometry.

    Returns
    -------
    :
        Specular reflectometry graph.
    """
    graph = {
        **beamline.beamline(scatter=True),
        **tof.elastic_wavelength("tof"),
        "theta": theta,
        "Q": reflectometry_q,
        "incident_beam": lambda: incident_beam,
        "sample_position": lambda: sample_position,
        "sample_rotation": lambda: sample_rotation,
        "detector_rotation": lambda: detector_rotation,
        "gravity": lambda: gravity,
    }
    return SpecularReflectionCoordTransformGraph(graph)


def add_coords(
    da: ReducibleDetectorData[RunType],
    graph: SpecularReflectionCoordTransformGraph[RunType],
) -> DataWithScatteringCoordinates[RunType]:
    da = da.transform_coords(
        ["theta", "wavelength", "Q", "detector_rotation"], graph=graph
    )
    da.coords.set_aligned('detector_rotation', False)
    da.coords["z_index"] = sc.arange(
        "row", 0, da.sizes["blade"] * da.sizes["wire"], unit=None
    ).fold("row", sizes={dim: da.sizes[dim] for dim in ("blade", "wire")})
    da.coords["y_index"] = sc.arange("stripe", 0, da.sizes["stripe"], unit=None)
    return da


def add_masks(
    da: DataWithScatteringCoordinates[RunType],
    ylim: YIndexLimits,
    wb: WavelengthBins,
    zlim: ZIndexLimits,
    bdlim: BeamDivergenceLimits,
) -> MaskedData[RunType]:
    da.masks["beam_divergence_too_large"] = (
        da.coords["angle_from_center_of_beam"] < bdlim[0]
    ) | (da.coords["angle_from_center_of_beam"] > bdlim[1])
    da.masks["y_index_range"] = (da.coords["y_index"] < ylim[0]) | (
        da.coords["y_index"] > ylim[1]
    )
    da.bins.masks["wavelength_mask"] = (da.bins.coords["wavelength"] < wb[0]) | (
        da.bins.coords["wavelength"] > wb[-1]
    )
    da.masks["z_index_range"] = (da.coords["z_index"] < zlim[0]) | (
        da.coords["z_index"] > zlim[1]
    )
    return da


providers = (
    add_masks,
    add_coords,
    specular_reflection,
)
