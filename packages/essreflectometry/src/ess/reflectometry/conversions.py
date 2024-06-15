# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.constants import h, m_n, pi
from scippneutron._utils import elem_dtype, elem_unit
from scippneutron.conversion.graph import beamline, tof

from .types import (
    ReducibleDetectorData,
    EventData,
    Gravity,
    IncidentBeam,
    MaskedEventData,
    RunType,
    SamplePosition,
    SampleRotation,
    SpecularReflectionCoordTransformGraph,
    WavelengthBins,
    YIndexLimits,
    ZIndexLimits,
)


def theta(
    gravity: sc.Variable,
    wavelength: sc.Variable,
    scattered_beam: sc.Variable,
    sample_rotation: sc.Variable,
) -> sc.Variable:
    """
    Compute the theta angle, including gravity correction,
    This is similar to the theta calculation in SANS (see
    https://docs.mantidproject.org/nightly/algorithms/Q1D-v2.html#q-unit-conversion),
    but we ignore the horizontal `x` component.
    See the schematic in Fig 5 of doi: 10.1016/j.nima.2016.03.007.

    Parameters
    ----------
    gravity:
        The three-dimensional vector describing gravity.
    wavelength:
        Wavelength values for the events.
    scatter_beam:
        Vector for scattered beam.
    sample_rotation:
        Rotation of sample wrt to incoming beam.

    Returns
    -------
    :
        Theta values, accounting for gravity.
    """
    grav = sc.norm(gravity)
    L2 = sc.norm(scattered_beam)
    y = sc.dot(scattered_beam, gravity) / grav
    y_correction = sc.to_unit(wavelength, elem_unit(L2), copy=True)
    y_correction *= y_correction
    drop = L2**2
    drop *= grav * (m_n**2 / (2 * h**2))
    # Optimization when handling either the dense or the event coord of binned data:
    # - For the event coord, both operands have same dims, and we can multiply in place
    # - For the dense coord, we need to broadcast using non in-place operation
    if set(drop.dims).issubset(set(y_correction.dims)):
        y_correction *= drop
    else:
        y_correction = y_correction * drop
    y_correction += y
    out = sc.abs(y_correction, out=y_correction)
    out /= L2
    out = sc.asin(out, out=out)
    out -= sc.to_unit(sample_rotation, "rad")
    return out


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
        "gravity": lambda: gravity,
    }
    return SpecularReflectionCoordTransformGraph(graph)


def add_coords(
    da: ReducibleDetectorData[RunType],
    graph: SpecularReflectionCoordTransformGraph[RunType],
) -> EventData[RunType]:
    da = da.transform_coords(["theta", "wavelength", "Q"], graph=graph)
    da.coords["z_index"] = sc.arange(
        "row", 0, da.sizes["blade"] * da.sizes["wire"], unit=None
    ).fold("row", sizes=dict(blade=da.sizes["blade"], wire=da.sizes["wire"]))
    da.coords["y_index"] = sc.arange("stripe", 0, da.sizes["stripe"], unit=None)
    return da


def add_masks(
    da: EventData[RunType], ylim: YIndexLimits, wb: WavelengthBins, zlim: ZIndexLimits
) -> MaskedEventData[RunType]:
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
