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
    reference.bins.masks['invalid'] = sc.isnan(R)
    reference /= R
    return reference.bins.concat(('stripe',)).hist(wavelength=wavelength_bins)


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
    h = reference.flatten(to='Q').hist(Q=qbins)
    R = sample.bins.concat().bin(Q=qbins) / h.data
    R.coords['Q_resolution'] = sc.sqrt(
        (
            (reference * reference.coords['Q_resolution'] ** 2)
            .flatten(to='Q')
            .hist(Q=qbins)
        )
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
    return sample.bins.concat(('stripe',)).bin(wavelength=wbins) / reference.data


providers = (
    reduce_reference,
    reduce_sample_over_q,
    reduce_sample_over_zw,
)
