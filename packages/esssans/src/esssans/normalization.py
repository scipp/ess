# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Dict, Optional

import scipp as sc
import scippneutron as scn

from .logging import get_logger
from .uncertainty import variance_normalized_signal_over_monitor
from .types import (
    CleanDirectBeam,
    IofQDenominator,
    TransmissionFraction,
    CleanMonitor,
    SampleRun,
    DirectRun,
    Incident,
    Transmission,
    SolidAngle,
    WavelengthData,
)


def solid_angle_of_rectangular_pixels(
    data: sc.DataArray, pixel_width: sc.Variable, pixel_height: sc.Variable
) -> sc.DataArray:
    """
    Solid angle computed from rectangular pixels with a 'width' and a 'height'.

    Note that this is an approximation which is only valid for small angles
    between the line of sight and the rectangle normal.

    Parameters
    ----------
    data:
        The DataArray that contains the positions for the detector pixels and the
        sample.
    pixel_width:
        The width of the rectangular pixels.
    pixel_height:
        The height of the rectangular pixels.

    Returns
    -------
    :
        The solid angle of the detector pixels, as viewed from the sample position.
        Any masks that have a dimension common to the dimensions of the position
        coordinate are retained to the output.
    """
    L2 = scn.L2(data)
    omega = (pixel_width * pixel_height) / (L2 * L2)
    solid_angle = sc.DataArray(data=omega)
    omega_dims = set(omega.dims)
    for key, mask in data.masks.items():
        if set(mask.dims).issubset(omega_dims):
            solid_angle.masks[key] = mask
    return solid_angle


def solid_angle_rectangular_approximation(
    data: WavelengthData[SampleRun],
) -> SolidAngle:
    return SolidAngle(
        solid_angle_of_rectangular_pixels(
            data,
            pixel_width=data.coords['pixel_width'],
            pixel_height=data.coords['pixel_height'],
        )
    )


def transmission_fraction(
    sample_incident_monitor: CleanMonitor[SampleRun, Incident],
    sample_transmission_monitor: CleanMonitor[SampleRun, Transmission],
    direct_incident_monitor: CleanMonitor[DirectRun, Incident],
    direct_transmission_monitor: CleanMonitor[DirectRun, Transmission],
) -> TransmissionFraction:
    """
    Approximation based on equations in
    [CalculateTransmission](https://docs.mantidproject.org/v4.0.0/algorithms/CalculateTransmission-v1.html)
    documentation:
    ``(Sample_T_monitor / Direct_T_monitor) * (Direct_I_monitor / Sample_I_monitor)``

    This is equivalent to ``mantid.CalculateTransmission`` without fitting.
    Inputs should be wavelength-dependent.

    TODO: It seems we are always multiplying this by data_monitors['incident'] to
    compute the normalization term. We could consider just returning
    ``(Sample_T_monitor / Direct_T_monitor) * Direct_I_monitor``

    Parameters
    ----------
    data_monitors:
        The data arrays for the incident and transmission monitors for the measurement
        run (monitor data should depend on wavelength).
    direct_monitors:
        The data arrays for the incident and transmission monitors for the direct
        run (monitor data should depend on wavelength).

    Returns
    -------
    :
        The transmission fraction computed from the monitor counts.
    """  # noqa: E501
    frac = (sample_transmission_monitor / direct_transmission_monitor) * (
        direct_incident_monitor / sample_incident_monitor
    )
    return TransmissionFraction(frac)


def _verify_normalization_alpha(
    numerator: sc.DataArray,
    denominator: sc.DataArray,
    signal_over_monitor_threshold: float = 0.1,
):
    """
    Verify that the ratio of sample detector counts to monitor counts is small, so
    we can safely drop the variances of the monitor to avoid broadcasting issues.
    See Heybrock et al. (2023).
    """
    alpha = variance_normalized_signal_over_monitor(numerator, denominator)
    if alpha > 0.25 * signal_over_monitor_threshold:
        logger = get_logger('sans')
        logger.warning(
            f'signal_over_monitor = {alpha} is close to the specified threshold of '
            f'{signal_over_monitor_threshold}. This means we are close to the regime '
            'where it is no longer safe to drop the variances of the normalization '
            'term.'
        )
    if alpha > signal_over_monitor_threshold:
        raise ValueError(
            f'signal_over_monitor = {alpha} > {signal_over_monitor_threshold}! '
            'This means that the ratio of detector counts to monitor counts is too '
            'high, and the variances of the monitor data cannot be safely dropped.'
        )


def iofq_denominator(
    # data: WavelengthData[SampleRun],
    data_transmission_monitor: CleanMonitor[SampleRun, Transmission],
    direct_incident_monitor: CleanMonitor[DirectRun, Incident],
    direct_transmission_monitor: CleanMonitor[DirectRun, Transmission],
    solid_angle: SolidAngle,
    direct_beam: Optional[CleanDirectBeam],
    # signal_over_monitor_threshold: float = 0.1,
) -> IofQDenominator:
    """
    Compute the denominator term for the I(Q) normalization. This is basically:
    ``solid_angle * direct_beam * data_transmission_monitor * direct_incident_monitor / direct_transmission_monitor``
    If the solid angle is not supplied, it is assumed to be 1.
    If the direct beam is not supplied, it is assumed to be 1.

    Because the multiplication between the wavelength dependent terms (monitor counts)
    and the pixel dependent term (solid angle) consists of a broadcast operation which
    would introduce correlations, we strip the data of variances.
    It is the responsibility of the user to ensure that the variances are small enough
    that they can be ignored. See more details in Heybrock et al. (2023).

    Parameters
    ----------
    data:
        The detector counts.
    data_transmission_monitor:
        The transmission monitor counts from the measurement run (depends on
        wavelength).
    direct_incident_monitor:
        The incident monitor counts from the direct run (depends on wavelength).
    direct_transmission_monitor:
        The transmission monitor counts from the direct run (depends on wavelength).
    direct_beam:
        The DataArray containing the direct beam function (depends on wavelength).
    signal_over_monitor_threshold:
        The threshold for the ratio of detector counts to monitor counts above which
        an error is raised because it is not safe to drop the variances of the monitor.

    Returns
    -------
    :
        The denominator for the SANS I(Q) normalization.
    """  # noqa: E501
    signal_over_monitor_threshold: float = (0.1,)
    denominator = (
        data_transmission_monitor.value
        * direct_incident_monitor.value
        / direct_transmission_monitor.value
    )
    if direct_beam is not None:
        denominator *= direct_beam

    # We need to remove the variances because the broadcasting operation between
    # solid_angle (pixel-dependent) and monitors (wavelength-dependent) will fail.
    # We check beforehand that the ratio of sample detector counts to monitor
    # counts is small
    # TODO move this into separate provider?
    # if denominator.variances is not None:
    #    _verify_normalization_alpha(
    #        numerator=data.hist(wavelength=denominator.coords['wavelength']),
    #        denominator=denominator,
    #        signal_over_monitor_threshold=signal_over_monitor_threshold,
    #    )

    denominator = solid_angle * sc.values(denominator)
    # Convert wavelength coordinate to midpoints for future histogramming
    # if wavelength_to_midpoints:
    denominator.coords['wavelength'] = sc.midpoints(denominator.coords['wavelength'])
    return IofQDenominator(denominator)


def normalize(numerator: sc.DataArray, denominator: sc.DataArray) -> sc.DataArray:
    """
    Perform normalization of counts as a function of Q.
    If the numerator contains events, we use the sc.lookup function to perform the
    division.

    Parameters
    ----------
    numerator:
        The data whose counts will be divided by the denominator. This can either be
        event or dense (histogrammed) data.
    denominator:
        The divisor for the normalization operation. This cannot be event data, it must
        contain histogrammed data.

    Returns
    -------
    :
        The input data normalized by the supplied denominator.
    """
    if numerator.bins is not None:
        return numerator.bins / sc.lookup(func=denominator, dim='Q')
    else:
        return numerator / denominator


providers = [
    transmission_fraction,
    iofq_denominator,
    solid_angle_rectangular_approximation,
]
