# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Optional

import scipp as sc
import scippneutron as scn
from scipp.core import concepts

from .types import (
    BackgroundRun,
    BackgroundTransmissionRun,
    CalibratedMaskedData,
    Clean,
    CleanDirectBeam,
    CleanMonitor,
    CleanSummedQ,
    Denominator,
    DirectRun,
    Incident,
    IofQ,
    Numerator,
    RunType,
    SampleRun,
    SampleTransmissionRun,
    SolidAngle,
    Transmission,
    TransmissionFraction,
    TransmissionFractionTimesDirectBeam,
    TransmissionRunType,
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
    sample_incident_monitor: CleanMonitor[TransmissionRunType, Incident],
    sample_transmission_monitor: CleanMonitor[TransmissionRunType, Transmission],
    direct_incident_monitor: CleanMonitor[DirectRun, Incident],
    direct_transmission_monitor: CleanMonitor[DirectRun, Transmission],
) -> TransmissionFraction[TransmissionRunType]:
    """
    Approximation based on equations in
    [CalculateTransmission](https://docs.mantidproject.org/v4.0.0/algorithms/CalculateTransmission-v1.html)
    documentation:
    ``(Sample_T_monitor / Direct_T_monitor) * (Direct_I_monitor / Sample_I_monitor)``

    This is equivalent to ``mantid.CalculateTransmission`` without fitting.
    Inputs should be wavelength-dependent.

    Parameters
    ----------
    data_monitors:
        The data arrays for the incident and transmission monitors for the transmission
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
    return TransmissionFraction[TransmissionRunType](frac)


_broadcasters = {
    UncertaintyBroadcastMode.drop: drop_variances_if_broadcast,
    UncertaintyBroadcastMode.upper_bound: broadcast_with_upper_bound_variances,
    UncertaintyBroadcastMode.fail: lambda x, sizes: x,
}


def transmission_fraction_times_direct_beam(
    transmission_fraction: TransmissionFraction[TransmissionRunType],
    direct_beam: Optional[CleanDirectBeam],
    uncertainties: UncertaintyBroadcastMode,
) -> TransmissionFractionTimesDirectBeam[TransmissionRunType]:
    """
    Compute the wavelength-dependen contribution to the denominator term for the I(Q) normalization.

    This is basically:
    ``transmission_fraction * direct_beam``
    If the direct beam is not supplied, it is assumed to be 1.

    Because the multiplication between the transmission fraction (pixel-independent)
    and the direct beam (potentially pixel-dependent) consists of a broadcast operation
    which would introduce correlations, variances of the direct beam are dropped or
    replaced by an upper-bound estimation, depending on the configured mode.

    Parameters
    ----------
    transmission_fraction:
        The transmission fraction (depends on wavelength).
    direct_beam:
        The DataArray containing the direct beam function (depends on wavelength).
    uncertainties:
        The mode for broadcasting uncertainties. See
        :py:class:`UncertaintyBroadcastMode` for details.

    Returns
    -------
    :
        Wavelength-dependent term transmission_fraction * direct_beam to be used for
        the denominator of the SANS I(Q) normalization.
        Used by :py:func:`iofq_denominator_sample` and
        :py:func:`iofq_denominator_background`.
    """
    out = transmission_fraction
    # We need to remove the variances because the broadcasting operation between
    # solid_angle (pixel-dependent) and monitors (wavelength-dependent) will fail.
    # The direct beam may also be pixel-dependent. In this case we need to drop
    # variances of the other term already before multiplying by solid_angle.
    if direct_beam is not None:
        broadcast = _broadcasters[uncertainties]
        # TODO: Do we need an additional check for the case where the direct beam
        # could be bin centers and the transmission fraction is bin edges? In that case
        # we would want to raise instead of broadcasting.
        out = direct_beam * broadcast(out, sizes=direct_beam.sizes)
    return TransmissionFractionTimesDirectBeam[TransmissionRunType](out)


def _iofq_denominator(
    incident_monitor: sc.DataArray,
    transmission_fraction_times_direct_beam: sc.DataArray,
    solid_angle: sc.DataArray,
    uncertainties: UncertaintyBroadcastMode,
) -> sc.DataArray:
    """
    Compute the denominator term for the I(Q) normalization.

    This is basically:
    ``(incident_monitor * direct_beam * transmission_fraction) * solid_angle``
    The `transmission_fraction_times_direct_beam` is computed by
    :py:func:`transmission_fraction_times_direct_beam`.

    Because the multiplication between the wavelength dependent terms
    and the pixel dependent term (solid angle) consists of a broadcast operation which
    would introduce correlations, variances are dropped or replaced by an upper-bound
    estimation, depending on the configured mode.


    Parameters
    ----------
    incident_monitor:
        The incident monitor data (depends on wavelength).
    transmission_fraction_times_direct_beam:
        The wavelength-dependent term for the denominator of the SANS I(Q)
        normalization. This is the output of
        :py:func:`transmission_fraction_times_direct_beam`.
    solid_angle:
        The solid angle of the detector pixels, as viewed from the sample position.
    uncertainties:
        The mode for broadcasting uncertainties. See
        :py:class:`UncertaintyBroadcastMode` for details.

    Returns
    -------
    :
        The denominator for the SANS I(Q) normalization.
    """  # noqa: E501
    broadcast = _broadcasters[uncertainties]
    wavelength_term = incident_monitor * transmission_fraction_times_direct_beam
    # Convert wavelength coordinate to midpoints for future histogramming
    wavelength_term.coords['wavelength'] = sc.midpoints(
        wavelength_term.coords['wavelength']
    )
    denominator = solid_angle * broadcast(
        wavelength_term,
        sizes=solid_angle.sizes,
    )
    return denominator


def iofq_denominator_sample(
    incident_monitor: CleanMonitor[SampleRun, Incident],
    transmission_fraction_times_direct_beam: TransmissionFractionTimesDirectBeam[
        SampleTransmissionRun
    ],
    solid_angle: SolidAngle[SampleRun],
    uncertainties: UncertaintyBroadcastMode,
) -> Clean[SampleRun, Denominator]:
    """
    Compute the denominator term for the I(Q) normalization for the sample run.
    """
    return Clean[SampleRun, Denominator](
        _iofq_denominator(
            incident_monitor,
            transmission_fraction_times_direct_beam,
            solid_angle,
            uncertainties,
        )
    )


def iofq_denominator_background(
    incident_monitor: CleanMonitor[BackgroundRun, Incident],
    transmission_fraction_times_direct_beam: TransmissionFractionTimesDirectBeam[
        BackgroundTransmissionRun
    ],
    solid_angle: SolidAngle[BackgroundRun],
    uncertainties: UncertaintyBroadcastMode,
) -> Clean[BackgroundRun, Denominator]:
    """
    Compute the denominator term for the I(Q) normalization for the background run.
    """
    return Clean[BackgroundRun, Denominator](
        _iofq_denominator(
            incident_monitor,
            transmission_fraction_times_direct_beam,
            solid_angle,
            uncertainties,
        )
    )


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
    transmission_fraction_times_direct_beam,
    iofq_denominator_sample,
    iofq_denominator_background,
    normalize,
    solid_angle_rectangular_approximation,
]
