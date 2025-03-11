# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
import scipp as sc

from ..reflectometry.conversions import reflectometry_q
from ..reflectometry.types import (
    BeamDivergenceLimits,
    CoordTransformationGraph,
    WavelengthBins,
    YIndexLimits,
    ZIndexLimits,
)
from .geometry import Detector
from .types import GravityToggle


def theta(wavelength, pixel_divergence_angle, L2, sample_rotation, detector_rotation):
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
    out = (c * L2 * wavelength**2).to(unit='dimensionless') + sc.sin(
        pixel_divergence_angle.to(unit='rad', copy=False)
        + detector_rotation.to(unit='rad')
    )
    out = sc.asin(out, out=out)
    out -= sample_rotation.to(unit='rad')
    return out


def theta_no_gravity(
    wavelength, pixel_divergence_angle, sample_rotation, detector_rotation
):
    '''
    Angle of reflection.

    Computes the angle between the scattering direction of
    the neutron and the sample surface while disregarding the
    effect of gravity.
    '''
    theta = (
        pixel_divergence_angle.to(unit='rad', copy=False)
        + detector_rotation.to(unit='rad')
        - sample_rotation.to(unit='rad')
    )
    if wavelength.bins:
        return sc.bins_like(wavelength, theta)
    return theta


def divergence_angle(theta, sample_rotation, detector_rotation):
    """
    Difference between the incident angle and the center of the incident beam.
    Useful for filtering parts of the beam that have too high divergence.

    On the Amor instrument this is always in the interval [-0.75 deg, 0.75 deg],
    but the divergence of the incident beam can be made lower.
    """
    return (
        theta.to(unit='rad', copy=False)
        - detector_rotation.to(unit='rad')
        + sample_rotation.to(unit='rad')
    )


def wavelength(
    event_time_offset, pixel_divergence_angle, L1, L2, chopper_phase, chopper_frequency
):
    "Converts event_time_offset to wavelength using the chopper settings."
    out = event_time_offset.to(unit="ns", dtype="float64", copy=True)
    unit = out.bins.unit
    tau = (1 / (2 * chopper_frequency.to(unit='Hz'))).to(unit=unit)
    tof_offset = tau * chopper_phase.to(unit='rad') / (np.pi * sc.units.rad)

    minimum = -tof_offset
    frame_bound = tau - tof_offset
    maximum = 2 * tau - tof_offset

    # Frame unwrapping
    out += sc.where(
        (minimum < event_time_offset) & (event_time_offset < frame_bound),
        tof_offset,
        sc.where(
            (frame_bound < event_time_offset) & (event_time_offset < maximum),
            tof_offset - tau,
            sc.scalar(np.nan, unit=unit),
        ),
    )
    # Correction for path length through guides being different
    # depending on incident angle.
    out -= (pixel_divergence_angle.to(unit="rad") / (np.pi * sc.units.rad)) * tau
    out *= (sc.constants.h / sc.constants.m_n) / (L1 + L2)
    return out.to(unit='angstrom', copy=False)


def coordinate_transformation_graph(gravity: GravityToggle) -> CoordTransformationGraph:
    return {
        "wavelength": wavelength,
        "theta": theta if gravity else theta_no_gravity,
        "divergence_angle": divergence_angle,
        "Q": reflectometry_q,
        "L1": lambda chopper_distance: sc.abs(chopper_distance),
        "L2": lambda distance_in_detector: distance_in_detector + Detector.distance,
    }


def _not_between(v, a, b):
    return (v < a) | (v > b)


def add_masks(
    da: sc.DataArray,
    ylim: YIndexLimits,
    zlims: ZIndexLimits,
    bdlim: BeamDivergenceLimits,
    wbins: WavelengthBins,
) -> sc.DataArray:
    """
    Masks the data by ranges in the detector
    coordinates ``z`` and ``y``, and by the divergence of the beam,
    and by wavelength.
    """
    da = da.assign_masks(
        stripe_range=_not_between(da.coords["stripe"], *ylim),
        z_range=_not_between(da.coords["z_index"], *zlims),
    )
    da = da.bins.assign_masks(
        divergence_too_large=_not_between(
            da.bins.coords["divergence_angle"],
            bdlim[0].to(unit=da.bins.coords["divergence_angle"].bins.unit),
            bdlim[1].to(unit=da.bins.coords["divergence_angle"].bins.unit),
        ),
        wavelength=_not_between(
            da.bins.coords['wavelength'],
            wbins[0],
            wbins[-1],
        ),
    )
    return da


providers = (coordinate_transformation_graph,)
