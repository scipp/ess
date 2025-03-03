# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry.tools import fwhm_to_std


def wavelength_resolution(
    # What parameters are needed?
):
    """
    Find the wavelength resolution contribution of the ESTIA instrument.

    Parameters
    ----------

    L1:
        Distance from midpoint between choppers to sample.
    L2:
        Distance from sample to detector.

    Returns
    -------
    :
        The wavelength resolution variable, as standard deviation.
    """
    # Don't yet know how to do this
    raise NotImplementedError()


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
    return fwhm_to_std(sample_size / L2.to(unit=sample_size.unit))


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
