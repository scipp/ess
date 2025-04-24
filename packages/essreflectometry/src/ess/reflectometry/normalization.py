# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc

from .supermirror import (
    Alpha,
    CriticalEdge,
    MValue,
    supermirror_reflectivity,
)
from .types import (
    QBins,
    ReducedReference,
    ReducibleData,
    Reference,
    ReferenceRun,
    ReflectivityOverQ,
    ReflectivityOverZW,
    Sample,
    WavelengthBins,
)


def reduce_reference(
    reference: ReducibleData[ReferenceRun],
    wavelength_bins: WavelengthBins,
    critical_edge: CriticalEdge,
    mvalue: MValue,
    alpha: Alpha,
) -> ReducedReference:
    """
    Reduces the reference measurement to estimate the
    intensity distribution in the detector for
    an ideal sample with reflectivity :math:`R = 1`.
    """
    R = supermirror_reflectivity(
        reference.bins.coords['Q'],
        c=critical_edge,
        m=mvalue,
        alpha=alpha,
    )
    reference = reference.bins.assign_masks(invalid=sc.isnan(R))
    reference = reference / R
    out = reference.bins.concat(('stripe',)).hist(wavelength=wavelength_bins)

    if 'position' in reference.coords:
        out.coords['position'] = reference.coords['position'].mean('stripe')
    return out


def reduce_sample_over_q(
    sample: Sample,
    reference: Reference,
    qbins: QBins,
) -> ReflectivityOverQ:
    """
    Computes reflectivity as ratio of
    sample intensity and intensity from a sample
    with ideal reflectivity.

    Returns reflectivity as a function of :math:`Q`.
    """
    s = sample.bins.concat().bin(Q=qbins)
    h = sc.values(
        (reference if reference.bins is None else reference.bins.concat()).hist(
            Q=s.coords['Q']
        )
    )
    R = s / h.data
    if 'Q_resolution' in reference.coords or 'Q_resolution' in reference.bins.coords:
        resolution = (
            reference.coords['Q_resolution']
            if 'Q_resolution' in reference.coords
            else reference.bins.coords['Q_resolution']
        )
        weighted_resolution = sc.values(reference) * resolution**2
        R.coords['Q_resolution'] = sc.sqrt(
            (
                weighted_resolution
                if weighted_resolution.bins is None
                else weighted_resolution.bins.concat()
            ).hist(Q=s.coords['Q'])
            / h
        ).data
    return R


def reduce_sample_over_zw(
    sample: Sample,
    reference: Reference,
    wbins: WavelengthBins,
) -> ReflectivityOverZW:
    """
    Computes reflectivity as ratio of
    sample intensity and intensity from a sample
    with ideal reflectivity.

    Returns reflectivity as a function of ``blade``, ``wire`` and :math:`\\wavelength`.
    """
    return sample.bins.concat(('stripe',)).bin(wavelength=wbins) / sc.values(
        reference.data
    )


providers = (
    reduce_reference,
    reduce_sample_over_q,
    reduce_sample_over_zw,
)
