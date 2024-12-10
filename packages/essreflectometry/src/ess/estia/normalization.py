import scipp as sc

from ..reflectometry.conversions import reflectometry_q
from ..reflectometry.supermirror import (
    Alpha,
    CriticalEdge,
    MValue,
    supermirror_reflectivity,
)
from ..reflectometry.types import (
    DetectorSpatialResolution,
    QBins,
    ReducedReference,
    ReducibleData,
    Reference,
    ReferenceRun,
    ReflectivityOverQ,
    ReflectivityOverZW,
    Sample,
    SampleRun,
    WavelengthBins,
)
from .conversions import theta
from .resolution import (
    angular_resolution,
    q_resolution,
    sample_size_resolution,
    wavelength_resolution,
)


def mask_events_where_supermirror_does_not_cover(
    sam: ReducibleData[SampleRun],
    ref: ReducedReference,
    critical_edge: CriticalEdge,
    mvalue: MValue,
    alpha: Alpha,
) -> Sample:
    """
    Mask events in regions of the detector the reference does not cover.

    Regions of the detector that the reference
    measurement doesn't cover cannot be used to compute reflectivity.

    Preferably the reference measurement should cover the entire
    detector, but sometimes that is not possible, for example
    if the supermirror :math:`M` value was too limited or because the reference
    was measured at too high angle.

    To figure out what events need to be masked,
    compute the supermirror reflectivity as a function
    of the :math:`Q` the event would have had if it had belonged to
    the reference measurement.
    """
    R = supermirror_reflectivity(
        reflectometry_q(
            sam.bins.coords["wavelength"],
            theta(
                sam.coords["pixel_divergence_angle"],
                ref.coords["sample_rotation"],
                ref.coords["detector_rotation"],
            ),
        ),
        c=critical_edge,
        m=mvalue,
        alpha=alpha,
    )
    sam.bins.masks["supermirror_does_not_cover"] = sc.isnan(R)
    return sam


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
    R = sample.bins.concat(('stripe',)).bin(wavelength=wbins) / reference.data
    R.masks["too_few_events"] = reference.data < sc.scalar(1, unit="counts")
    return R


def evaluate_reference(
    reference: ReducedReference,
    sample: ReducibleData[SampleRun],
    qbins: QBins,
    detector_spatial_resolution: DetectorSpatialResolution[SampleRun],
) -> Reference:
    """
    Adds a :math:`Q` and :math:`Q`-resolution coordinate to each bin of the ideal
    intensity distribution. The coordinates are computed as if the data came from
    the sample measurement, that is, they use the ``sample_rotation``
    and ``detector_rotation`` parameters from the sample measurement.
    """
    ref = reference.copy()
    ref.coords["sample_rotation"] = sample.coords["sample_rotation"]
    ref.coords["detector_rotation"] = sample.coords["detector_rotation"]
    ref.coords["sample_size"] = sample.coords["sample_size"]
    ref.coords["detector_spatial_resolution"] = detector_spatial_resolution
    ref.coords["wavelength"] = sc.midpoints(ref.coords["wavelength"])
    ref = ref.transform_coords(
        (
            "Q",
            "wavelength_resolution",
            "sample_size_resolution",
            "angular_resolution",
            "Q_resolution",
        ),
        {
            "divergence_angle": "pixel_divergence_angle",
            "theta": theta,
            "Q": reflectometry_q,
            "wavelength_resolution": wavelength_resolution,
            "sample_size_resolution": sample_size_resolution,
            "angular_resolution": angular_resolution,
            "Q_resolution": q_resolution,
        },
        rename_dims=False,
    )
    return sc.values(ref)


providers = (
    reduce_reference,
    reduce_sample_over_q,
    reduce_sample_over_zw,
    evaluate_reference,
    mask_events_where_supermirror_does_not_cover,
)
