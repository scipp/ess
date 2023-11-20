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
    NormWavelengthTerm,
    Numerator,
    RunType,
    SampleRun,
    SampleTransmissionRun,
    SolidAngle,
    Transmission,
    TransmissionFraction,
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
    `CalculateTransmission <https://docs.mantidproject.org/v4.0.0/algorithms/CalculateTransmission-v1.html>`_
    documentation:
    ``(sample_transmission_monitor / direct_transmission_monitor) * (direct_incident_monitor / sample_incident_monitor)``

    This is equivalent to ``mantid.CalculateTransmission`` without fitting.
    Inputs should be wavelength-dependent.

    Parameters
    ----------
    sample_incident_monitor:
        The incident monitor data for the sample (transmission) run.
    sample_transmission_monitor:
        The transmission monitor data for the sample (transmission) run.
    direct_incident_monitor:
        The incident monitor data for the direct beam run.
    direct_transmission_monitor:
        The transmission monitor data for the direct beam run.

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


def _iofq_norm_wavelength_term(
    incident_monitor: sc.DataArray,
    transmission_fraction: sc.DataArray,
    direct_beam: Optional[sc.DataArray],
    uncertainties: UncertaintyBroadcastMode,
) -> sc.DataArray:
    """
    Compute the wavelength-dependent contribution to the denominator term for the I(Q)
    normalization.

    This is basically:
    ``incident_monitor * transmission_fraction * direct_beam``
    If the direct beam is not supplied, it is assumed to be 1.

    Because the multiplication between the ``incident_monitor * transmission_fraction``
    (pixel-independent) and the direct beam (potentially pixel-dependent) consists of a
    broadcast operation which would introduce correlations, variances of the direct
    beam are dropped or replaced by an upper-bound estimation, depending on the
    configured mode.

    Parameters
    ----------
    incident_monitor:
        The incident monitor data (depends on wavelength).
    transmission_fraction:
        The transmission fraction (depends on wavelength).
    direct_beam:
        The direct beam function (depends on wavelength).
    uncertainties:
        The mode for broadcasting uncertainties. See
        :py:class:`UncertaintyBroadcastMode` for details.

    Returns
    -------
    :
        Wavelength-dependent term
        (incident_monitor * transmission_fraction * direct_beam) to be used for
        the denominator of the SANS I(Q) normalization.
        Used by :py:func:`iofq_denominator`.
    """
    out = incident_monitor * transmission_fraction
    if direct_beam is not None:
        broadcast = _broadcasters[uncertainties]
        # TODO: Do we need an additional check for the case where the direct beam
        # could be bin centers and the transmission fraction is bin edges? In that case
        # we would want to raise instead of broadcasting.
        out = direct_beam * broadcast(out, sizes=direct_beam.sizes)
    # Convert wavelength coordinate to midpoints for future histogramming
    out.coords['wavelength'] = sc.midpoints(out.coords['wavelength'])
    return out


def iofq_norm_wavelength_term_sample(
    incident_monitor: CleanMonitor[SampleRun, Incident],
    transmission_fraction: TransmissionFraction[SampleTransmissionRun],
    direct_beam: Optional[CleanDirectBeam],
    uncertainties: UncertaintyBroadcastMode,
) -> NormWavelengthTerm[SampleRun]:
    """
    Compute the wavelength-dependent contribution to the denominator term for the I(Q)
    normalization, for the sample run.
    """
    return NormWavelengthTerm[SampleRun](
        _iofq_norm_wavelength_term(
            incident_monitor=incident_monitor,
            transmission_fraction=transmission_fraction,
            direct_beam=direct_beam,
            uncertainties=uncertainties,
        )
    )


def iofq_norm_wavelength_term_background(
    incident_monitor: CleanMonitor[BackgroundRun, Incident],
    transmission_fraction: TransmissionFraction[BackgroundTransmissionRun],
    direct_beam: Optional[CleanDirectBeam],
    uncertainties: UncertaintyBroadcastMode,
) -> NormWavelengthTerm[BackgroundRun]:
    """
    Compute the wavelength-dependent contribution to the denominator term for the I(Q)
    normalization, for the background run.
    """
    return NormWavelengthTerm[BackgroundRun](
        _iofq_norm_wavelength_term(
            incident_monitor=incident_monitor,
            transmission_fraction=transmission_fraction,
            direct_beam=direct_beam,
            uncertainties=uncertainties,
        )
    )


def iofq_denominator(
    wavelength_term: NormWavelengthTerm[RunType],
    solid_angle: SolidAngle[RunType],
    uncertainties: UncertaintyBroadcastMode,
) -> Clean[RunType, Denominator]:
    """
    Compute the denominator term for the I(Q) normalization.

    In a SANS experiment, the scattering cross section :math:`I(Q)` is defined as
    (`Heenan et al. 1997 <https://doi.org/10.1107/S0021889897002173>`_):

    .. math::

       I(Q) = \\frac{\\partial\\Sigma{Q}}{\\partial\\Omega} = \\frac{A_{H} \\Sigma_{R,\\lambda\\subset Q} C(R, \\lambda)}{A_{M} t \\Sigma_{R,\\lambda\\subset Q}M(\\lambda)T(\\lambda)D(\\lambda)\\Omega(R)}

    where :math:`A_{H}` is the area of a mask (which avoids saturating the detector)
    placed between the monitor of area :math:`A_{M}` and the main detector.
    :math:`\\Omega` is the detector solid angle, and :math:`C` is the count rate on the
    main detector, which depends on the position :math:`R` and the wavelength.
    :math:`t` is the sample thickness, :math:`M` represents the incident monitor count
    rate for the sample run, and :math:`T` is known as the transmission fraction.

    Note that the incident monitor used to compute the transmission fraction is not
    necessarily the same as :math:`M`, as the transmission fraction is usually computed
    from a separate 'transmission' run (in the 'sample' run, the transmission monitor is
    commonly moved out of the beam path, to avoid polluting the sample detector signal).

    Finally, :math:`D` is the 'direct beam function', and is defined as

    .. math::

       D(\\lambda) = \\frac{\\eta(\\lambda)}{\\eta_{M}(\\lambda)} \\frac{A_{H}}{A_{M}}

    where :math:`\\eta` and :math:`\\eta_{M}` are the detector and monitor
    efficiencies, respectively.

    Hence, in order to normalize the main detector counts :math:`C`, we need compute the
    transmission fraction :math:`T(\\lambda)`, the direct beam function
    :math:`D(\\lambda)` and the solid angle :math:`\\Omega(R)`.

    The denominator is then simply:
    :math:`M_{\\lambda} T_{\\lambda} D_{\\lambda} \\Omega_{R}`,
    which is equivalent to ``wavelength_term * solid_angle``.
    The ``wavelength_term`` includes all but the ``solid_angle`` and is computed by
    :py:func:`iofq_norm_wavelength_term_sample` or
    :py:func:`iofq_norm_wavelength_term_background`.

    Because the multiplication between the wavelength dependent terms
    and the pixel dependent term (solid angle) consists of a broadcast operation which
    would introduce correlations, variances are dropped or replaced by an upper-bound
    estimation, depending on the configured mode.

    Parameters
    ----------
    wavelength_term:
        The term that depends on wavelength, computed by
        :py:func:`iofq_norm_wavelength_term`.
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


providers = (
    transmission_fraction,
    iofq_norm_wavelength_term_sample,
    iofq_norm_wavelength_term_background,
    iofq_denominator,
    normalize,
    solid_angle_rectangular_approximation,
)
