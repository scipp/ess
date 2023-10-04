# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Optional

import scipp as sc
import scippneutron as scn
from scipp.core import concepts

from .types import (
    CalibratedMaskedData,
    Clean,
    CleanDirectBeam,
    CleanMonitor,
    CleanSummedQ,
    Denominator,
    DirectRun,
    Incident,
    IofQ,
    NormWavelengthTerm,
    Numerator,
    RunType,
    SampleRun,
    SolidAngle,
    Transmission,
    TransmissionFraction,
    UncertaintyBroadcastMode,
)
from .uncertainty import (
    broadcast_with_upper_bound_variances,
    drop_variances_if_broadcast,
)


def solid_angle_rectangular_approximation(
    data: CalibratedMaskedData[RunType],
) -> SolidAngle[RunType]:
    """
    Solid angle computed from rectangular pixels with a 'width' and a 'height'.

    Note that this is an approximation which is only valid for small angles
    between the line of sight and the rectangle normal.

    Parameters
    ----------
    data:
        The DataArray that contains the positions for the detector pixels and the
        sample, as well as `pixel_width` and `pixel_height` as coordinates.

    Returns
    -------
    :
        The solid angle of the detector pixels, as viewed from the sample position.
        Any coords and masks that have a dimension common to the dimensions of the
        position coordinate are retained in the output.
    """
    pixel_width = data.coords['pixel_width']
    pixel_height = data.coords['pixel_height']
    L2 = scn.L2(data)
    omega = (pixel_width * pixel_height) / (L2 * L2)
    dims = set(data.dims) - set(omega.dims)
    return SolidAngle[RunType](
        concepts.rewrap_reduced_data(prototype=data, data=omega, dim=dims)
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


_broadcasters = {
    UncertaintyBroadcastMode.drop: drop_variances_if_broadcast,
    UncertaintyBroadcastMode.upper_bound: broadcast_with_upper_bound_variances,
    UncertaintyBroadcastMode.fail: lambda x, sizes: x,
}


def iofq_norm_wavelength_term(
    data_transmission_monitor: CleanMonitor[RunType, Transmission],
    direct_incident_monitor: CleanMonitor[DirectRun, Incident],
    direct_transmission_monitor: CleanMonitor[DirectRun, Transmission],
    direct_beam: Optional[CleanDirectBeam],
    uncertainties: UncertaintyBroadcastMode,
) -> NormWavelengthTerm[RunType]:
    """
    Compute the wavelength-dependen contribution to the denominator term for the I(Q) normalization.

    This is basically:
    ``direct_beam * data_transmission_monitor * direct_incident_monitor / direct_transmission_monitor``
    If the direct beam is not supplied, it is assumed to be 1.

    Because the multiplication between the wavelength dependent terms (monitor counts)
    and the pixel dependent term (solid angle) consists of a broadcast operation which
    would introduce correlations, variances are dropped or replaced by an upper-bound
    estimation, depending on the configured mode.

    Parameters
    ----------
    data_transmission_monitor:
        The transmission monitor counts from the measurement run (depends on
        wavelength).
    direct_incident_monitor:
        The incident monitor counts from the direct run (depends on wavelength).
    direct_transmission_monitor:
        The transmission monitor counts from the direct run (depends on wavelength).
    direct_beam:
        The DataArray containing the direct beam function (depends on wavelength).
    uncertainties:
        The mode for broadcasting uncertainties. See
        :py:class:`UncertaintyBroadcastMode` for details.

    Returns
    -------
    :
        Wavelength-dependent term for the denominator of the SANS I(Q) normalization.
        Used by :py:func:`iofq_denominator`.
    """  # noqa: E501
    denominator = (
        data_transmission_monitor
        * direct_incident_monitor
        / direct_transmission_monitor
    )

    # We need to remove the variances because the broadcasting operation between
    # solid_angle (pixel-dependent) and monitors (wavelength-dependent) will fail.
    # The direct beam may also be pixel-dependent. In this case we need to drop
    # variances of the other term already before multiplying by solid_angle.
    if direct_beam is not None:
        broadcast = _broadcasters[uncertainties]
        denominator = direct_beam * broadcast(denominator, sizes=direct_beam.sizes)
    # Convert wavelength coordinate to midpoints for future histogramming
    # if wavelength_to_midpoints:
    denominator.coords['wavelength'] = sc.midpoints(denominator.coords['wavelength'])
    return NormWavelengthTerm[RunType](denominator)


def iofq_denominator(
    wavelength_term: NormWavelengthTerm[RunType],
    solid_angle: SolidAngle[RunType],
    uncertainties: UncertaintyBroadcastMode,
) -> Clean[RunType, Denominator]:
    """
    Compute the denominator term for the I(Q) normalization.

    This is basically:
    ``solid_angle * direct_beam * data_transmission_monitor * direct_incident_monitor / direct_transmission_monitor``
    The `wavelength_term` included all but the `solid_angle` and is computed by
    :py:func:`iofq_norm_wavelength_term`.

    Because the multiplication between the wavelength dependent terms (monitor counts)
    and the pixel dependent term (solid angle) consists of a broadcast operation which
    would introduce correlations, variances are dropped or replaced by an upper-bound
    estimation, depending on the configured mode.


    Parameters
    ----------
    solid_angle:
        The solid angle of the detector pixels, as viewed from the sample position.
    wavelength_term:
        The term that depends on wavelength, computed by :py:func:`iofq_norm_wavelength_term`.
    uncertainties:
        The mode for broadcasting uncertainties. See
        :py:class:`UncertaintyBroadcastMode` for details.

    Returns
    -------
    :
        The denominator for the SANS I(Q) normalization.
    """  # noqa: E501
    broadcast = _broadcasters[uncertainties]
    denominator = solid_angle * broadcast(wavelength_term, sizes=solid_angle.sizes)
    return Clean[RunType, Denominator](denominator)


def normalize(
    numerator: CleanSummedQ[RunType, Numerator],
    denominator: CleanSummedQ[RunType, Denominator],
) -> IofQ[RunType]:
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
    if denominator.variances is not None and numerator.bins is not None:
        # Event-mode normalization is not correct of norm-term has variances.
        # See https://doi.org/10.3233/JNR-220049 for context.
        numerator = numerator.hist()
    if numerator.bins is not None:
        da = numerator.bins / sc.lookup(func=denominator, dim='Q')
    else:
        da = numerator / denominator
    return IofQ[RunType](da)


providers = [
    transmission_fraction,
    iofq_norm_wavelength_term,
    iofq_denominator,
    normalize,
    solid_angle_rectangular_approximation,
]
