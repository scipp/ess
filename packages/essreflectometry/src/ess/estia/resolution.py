# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp import constants

from ..reflectometry.tools import fwhm_to_std


def wavelength_resolution(
    source_pulse_length: sc.Variable,
    wavelength: sc.Variable,
    Ltotal: sc.Variable,
):
    """
    Find the wavelength resolution contribution of the ESTIA instrument.

    Parameters
    ----------

    source_pulse_length:
        The length of the ESS source pulse.
    wavelength:
        The wavelength of the neutron.
    Ltotal:
        The distance from source to detector.

    Returns
    -------
    :
        The wavelength resolution,
        ratio of the standard deviation of wavelength and the wavelength.
    """
    # The exact factor depends on the shape of the source pulse.
    # The factor here assumes a rectangular source pulse:
    # import sympy as sp
    # x, D  = sp.symbols('x, D', positive=True)
    # sp.sqrt(sp.integrate(1/D * (x-D/2)**2, (x, 0, D)))
    standard_deviation_of_time_at_source = (1 / 12**0.5) * source_pulse_length
    tof = wavelength * Ltotal * constants.m_n / constants.h
    return (standard_deviation_of_time_at_source / tof).to(unit='dimensionless')


def sample_size_resolution(
    L2: sc.Variable,
    sample_size: sc.Variable,
):
    """
    The resolution from the projected sample size, where it may be bigger
    than the detector pixel resolution as described in Section 4.3.3 of the Amor
    publication (doi: 10.1016/j.nima.2016.03.007).

    Parameters
    ----------
    L2:
        Distance from sample to detector.
    sample_size:
        Size of sample.

    Returns
    -------
    :
        Standard deviation of contribution from the sample size.
    """
    return fwhm_to_std(sample_size.to(unit=L2.unit, dtype='float64') / L2)


def angular_resolution(
    theta: sc.Variable,
    L2: sc.Variable,
    detector_spatial_resolution: sc.Variable,
):
    """
    Determine the angular resolution of the ESTIA instrument.

    Parameters
    ----------
    theta:
        Angle of reflection.
    L2:
        Distance between sample and detector.
    detector_spatial_resolution:
        FWHM of detector pixel resolution.

    Returns
    -------
    :
        Angular resolution standard deviation over the reflection angle.
    """
    return (
        fwhm_to_std(
            sc.atan(detector_spatial_resolution.to(unit=L2.unit, dtype='float64') / L2)
        ).to(unit=theta.unit, dtype='float64')
        / theta
    )


def q_resolution(
    Q: sc.Variable,
    angular_resolution: sc.Variable,
    wavelength_resolution: sc.Variable,
    sample_size_resolution: sc.Variable,
):
    """
    Compute resolution in Q.

    Parameters
    ----------
    Q:
        Momentum transfer.
    angular_resolution:
        Angular resolution contribution.
    wavelength_resolution:
        Wavelength resolution contribution.
    sample_size_resolution:
        Sample size resolution contribution.

    Returns
    -------
    :
        Q resolution function.
    """
    return sc.sqrt(
        (angular_resolution**2 + wavelength_resolution**2 + sample_size_resolution**2)
        * Q**2
    )
