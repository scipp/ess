# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry.tools import fwhm_to_std


def wavelength_resolution(
    L1,
    L2,
    chopper_separation,
):
    """
    Find the wavelength resolution contribution as described in Section 4.3.3 of the
    Amor publication (doi: 10.1016/j.nima.2016.03.007).

    Parameters
    ----------

    L1:
        Distance from midpoint between choppers to sample.
    L2:
        Distance from sample to detector.
    chopper_separation:
        Distance between choppers.

    Returns
    -------
    :
        The wavelength resolution variable, as standard deviation.
    """
    return fwhm_to_std(sc.abs(chopper_separation) / (L1 + L2))


def sample_size_resolution(
    L2,
    sample_size,
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
    return fwhm_to_std(sample_size / L2.to(unit=sample_size.unit))


def angular_resolution(
    theta,
    L2,
    detector_spatial_resolution,
):
    """
    Determine the angular resolution as described in Section 4.3.3 of the Amor
    publication (doi: 10.1016/j.nima.2016.03.007).

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
        Angular resolution standard deviation
    """
    return (
        fwhm_to_std(
            sc.atan(
                detector_spatial_resolution
                / L2.to(unit=detector_spatial_resolution.unit)
            )
        ).to(unit=theta.unit)
        / theta
    )


def q_resolution(
    Q,
    angular_resolution,
    wavelength_resolution,
    sample_size_resolution,
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
