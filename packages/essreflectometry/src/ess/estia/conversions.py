# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry.conversions import reflectometry_q
from ..reflectometry.types import (
    BeamDivergenceLimits,
    CoordTransformationGraph,
    WavelengthBins,
    YIndexLimits,
    ZIndexLimits,
)


def theta(detector_position_relative_sample, sample_rotation):
    '''
    Angle of reflection.

    Computes the angle between the scattering direction of
    the neutron and the sample surface.

    Assumes that the sample is oriented almost parallel to yz plane
    but rotated around the y-axis by :code:`sample_rotation`.
    '''
    p = detector_position_relative_sample
    # Normal of yz plane.
    n = sc.vector([1.0, 0, 0], unit='dimensionless')
    np = sc.norm(p)
    pp = p - sc.dot(p, n) * n
    npp = sc.norm(pp)
    angle_to_zy_plane = sc.acos(sc.dot(p, pp) / (np * npp))
    return angle_to_zy_plane - sample_rotation.to(unit='rad')


def angle_of_divergence(
    theta, sample_rotation, angle_to_center_of_beam, natural_incidence_angle
):
    """
    Difference between the incident angle and the center of the incident beam.
    Useful for filtering parts of the beam that have too high divergence.

    This is always in the interval [-0.75 deg, 0.75 deg],
    but the divergence of the incident beam can also be reduced.
    """
    return theta - sample_rotation - angle_to_center_of_beam.to(unit='rad')


def wavelength(
    event_time_offset,
    # Other inputs
):
    "Converts event_time_offset to wavelength"
    # Use frame unwrapping from scippneutron
    pass


def coordinate_transformation_graph() -> CoordTransformationGraph:
    return {
        "detector_position_relative_sample": (
            lambda detector, sample: detector.position - sample.position
        ),
        "wavelength": wavelength,
        "theta": theta,
        "angle_of_divergence": angle_of_divergence,
        "Q": reflectometry_q,
        "L1": lambda source, sample: sample.position - source.position,
        "L2": lambda detector_position_relative_sample: sc.norm(
            detector_position_relative_sample
        ),
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
