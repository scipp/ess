# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.constants import pi

from ..reflectometry.conversions import reflectometry_q
from ..reflectometry.types import (
    BeamDivergenceLimits,
    CoordTransformationGraph,
    WavelengthBins,
    YIndexLimits,
    ZIndexLimits,
)


def theta(divergence_angle, sample_rotation, detector_rotation):
    '''
    Angle of reflection.

    Computes the angle between the scattering direction of
    the neutron and the sample surface.
    '''
    return (
        divergence_angle
        + detector_rotation.to(unit='rad')
        - sample_rotation.to(unit='rad')
    )


def angle_of_divergence(
    position,
    sample_position,
    detector_rotation,
):
    """
    Angle between the scattering direction and
    the ray from the sample to the center of the detector.
    """
    p = position - sample_position.to(unit=position.unit)
    # Normal to plane of zero sample rotation.
    n = sc.vector([1.0, 0, 0], unit='dimensionless')
    angle_to_yz_plane = pi / 2 * sc.scalar(1, unit='rad') - sc.acos(
        sc.dot(p, n) / sc.norm(p)
    )
    return angle_to_yz_plane - detector_rotation


def wavelength(
    event_time_offset,
    # Other inputs
):
    "Converts event_time_offset to wavelength"
    # Use frame unwrapping from scippneutron
    pass


def coordinate_transformation_graph() -> CoordTransformationGraph:
    return {
        "wavelength": wavelength,
        "theta": theta,
        "angle_of_divergence": angle_of_divergence,
        "Q": reflectometry_q,
        "L1": lambda source_position, sample_position: sc.norm(
            sample_position - source_position
        ),  # + extra correction for guides?
        "L2": lambda position, sample_position: sc.norm(position - sample_position),
    }


def add_coords(
    da: sc.DataArray,
    graph: dict,
) -> sc.DataArray:
    "Adds scattering coordinates to the raw detector data."
    return da.transform_coords(
        ("wavelength", "theta", "angle_of_divergence", "Q", "L1", "L2"),
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
