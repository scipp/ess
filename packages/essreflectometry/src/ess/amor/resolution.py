# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry.tools import fwhm_to_std
from ..reflectometry.types import (
    DetectorSpatialResolution,
    FootprintCorrectedData,
    QBins,
    QResolution,
    SampleRun,
    SampleSize,
)
from .types import (
    AngularResolution,
    Chopper1Position,
    Chopper2Position,
    SampleSizeResolution,
    WavelengthResolution,
)


def wavelength_resolution(
    da: FootprintCorrectedData[SampleRun],
    chopper_1_position: Chopper1Position[SampleRun],
    chopper_2_position: Chopper2Position[SampleRun],
) -> WavelengthResolution:
    """
    Find the wavelength resolution contribution as described in Section 4.3.3 of the
    Amor publication (doi: 10.1016/j.nima.2016.03.007).

    Parameters
    ----------
    chopper_1_position:
        Position of first chopper (the one closer to the source).
    chopper_2_position:
        Position of second chopper (the one closer to the sample).
    pixel_position:
        Positions for detector pixels.

    Returns
    -------
    :
        The angular resolution variable, as standard deviation.
    """
    pixel_position = da.coords["position"]
    distance_between_choppers = (
        chopper_2_position.fields.z - chopper_1_position.fields.z
    )
    chopper_midpoint = (chopper_1_position + chopper_2_position) * sc.scalar(0.5)
    chopper_detector_distance = pixel_position.fields.z - chopper_midpoint.fields.z
    return WavelengthResolution(
        fwhm_to_std(distance_between_choppers / chopper_detector_distance)
    )


def sample_size_resolution(
    da: FootprintCorrectedData[SampleRun],
    sample_size: SampleSize[SampleRun],
) -> SampleSizeResolution:
    """
    The resolution from the projected sample size, where it may be bigger
    than the detector pixel resolution as described in Section 4.3.3 of the Amor
    publication (doi: 10.1016/j.nima.2016.03.007).

    Parameters
    ----------
    pixel_position:
        Positions for detector pixels.
    sample_size:
        Size of sample.

    Returns
    -------
    :
        Standard deviation of contribution from the sample size.
    """
    return fwhm_to_std(
        sc.to_unit(sample_size, "m")
        / sc.to_unit(da.coords["position"].fields.z, "m", copy=False)
    )


def angular_resolution(
    da: FootprintCorrectedData[SampleRun],
    detector_spatial_resolution: DetectorSpatialResolution[SampleRun],
) -> AngularResolution:
    """
    Determine the angular resolution as described in Section 4.3.3 of the Amor
    publication (doi: 10.1016/j.nima.2016.03.007).

    Parameters
    ----------
    pixel_position:
        Positions for detector pixels.
    theta:
        Theta values for events.
    detector_spatial_resolution:
        FWHM of detector pixel resolution.

    Returns
    -------
    :
        Angular resolution standard deviation
    """
    theta = da.bins.coords["theta"]
    return (
        fwhm_to_std(
            sc.to_unit(
                sc.atan(
                    sc.to_unit(detector_spatial_resolution, "m")
                    / sc.to_unit(da.coords["position"].fields.z, "m", copy=False)
                ),
                theta.bins.unit,
                copy=False,
            )
        )
        / theta
    )


def sigma_Q(
    angular_resolution: AngularResolution,
    wavelength_resolution: WavelengthResolution,
    sample_size_resolution: SampleSizeResolution,
    q_bins: QBins,
) -> QResolution:
    """
    Combine all of the components of the resolution and add Q contribution.

    Parameters
    ----------
    angular_resolution:
        Angular resolution contribution.
    wavelength_resolution:
        Wavelength resolution contribution.
    sample_size_resolution:
        Sample size resolution contribution.
    q_bins:
        Q-bin values.

    Returns
    -------
    :
        Combined resolution function.
    """
    return sc.sqrt(
        angular_resolution**2 + wavelength_resolution**2 + sample_size_resolution**2
    ).flatten(to="detector_number").max("detector_number") * sc.midpoints(q_bins)


providers = (sigma_Q, angular_resolution, wavelength_resolution, sample_size_resolution)
