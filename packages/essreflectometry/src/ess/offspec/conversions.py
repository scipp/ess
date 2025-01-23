# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scippneutron.conversion.graph import beamline, tof

from ..reflectometry.conversions import reflectometry_q
from ..reflectometry.types import (
    BeamDivergenceLimits,
    CoordTransformationGraph,
    WavelengthBins,
    YIndexLimits,
    ZIndexLimits,
)


def theta(wavelength, scattered_beam, sample_rotation):
    '''
    Angle of reflection.

    Computes the angle between the scattering direction of
    the neutron and the sample surface.

    :math:`\\gamma^*` denotes the angle between the scattering direction
    and the horizontal plane.
    :math:`\\gamma` denotes the angle between the ray from sample position
    to detection position
    and the horizontal plane.
    :math:`L_2` is the length of the ray from sample position to detector position.
    :math:`v` is the velocity of the neutron at the sample.
    :math:`t` is the travel time from sample to detector.

    The parabolic trajectory of the neutron satisfies

    .. math::

        \\sin(\\gamma) L_2 = \\sin(\\gamma^*) v t - \\frac{g}{2} t^2

    and

    .. math::

        \\cos(\\gamma) L_2 = \\cos(\\gamma^*) vt

    where :math:`g` is the gravitational acceleration.

    The second equation tells us that the approximation :math:`L_2=vt`
    will have a small error if :math:`\\gamma` is close to 0 and
    the difference between :math:`\\gamma` and :math:`\\gamma^*` is small.

    Using this approximation we can solve the first equation,
    and by expressing :math:`v` in terms of the wavelength we get

    .. math::

        \\sin(\\gamma^*) =
        \\sin(\\gamma) + \\frac{g}{2} \\frac{L_2 \\lambda^2 h^2}{m_n^2}.

    Finally, the scattering angle is obtained by subtracting the sample rotation
    relative to the horizontal plane.
    '''
    c = sc.constants.g * sc.constants.m_n**2 / sc.constants.h**2
    L2 = sc.norm(scattered_beam)
    out = (c * L2 * wavelength**2).to(
        unit='dimensionless'
    ) + scattered_beam.fields.y / L2
    out = sc.asin(out, out=out)
    out -= sample_rotation.to(unit='rad')
    return out


def coordinate_transformation_graph() -> CoordTransformationGraph:
    return {
        **beamline.beamline(scatter=True),
        **tof.elastic_wavelength("tof"),
        "theta": theta,
        "Q": reflectometry_q,
    }


def add_coords(
    da: sc.DataArray,
    graph: dict,
) -> sc.DataArray:
    "Adds scattering coordinates to the raw detector data."
    return da.transform_coords(
        ("wavelength", "theta", "Q", "L1", "L2"),
        graph,
        rename_dims=False,
        keep_intermediate=False,
        keep_aliases=False,
    )


def _not_between(v, a, b):
    return (v < a) | (v > b)


def add_masks(
    da: sc.DataArray,
    ylim: YIndexLimits,
    zlims: ZIndexLimits,
    bdlim: BeamDivergenceLimits,
    wbins: WavelengthBins,
):
    """
    Masks the data by ranges in the detector
    coordinates ``z`` and ``y``, and by the divergence of the beam,
    and by wavelength.
    """
    da.masks["stripe_range"] = _not_between(da.coords["stripe"], *ylim)
    da.masks['z_range'] = _not_between(da.coords["z_index"], *zlims)
    da.bins.masks["divergence_too_large"] = _not_between(
        da.bins.coords["angle_of_divergence"],
        bdlim[0].to(unit=da.bins.coords["angle_of_divergence"].bins.unit),
        bdlim[1].to(unit=da.bins.coords["angle_of_divergence"].bins.unit),
    )
    da.bins.masks['wavelength'] = _not_between(
        da.bins.coords['wavelength'],
        wbins[0],
        wbins[-1],
    )
    return da


providers = (coordinate_transformation_graph,)
