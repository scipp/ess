# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry.conversions import reflectometry_q
from ..reflectometry.corrections import correct_by_footprint
from ..reflectometry.normalization import reduce_from_events_to_lz, reduce_reference
from ..reflectometry.supermirror import (
    Alpha,
    CriticalEdge,
    MValue,
    supermirror_reflectivity,
)
from ..reflectometry.types import (
    CoordTransformationGraph,
    DetectorSpatialResolution,
    ProtonCharge,
    ReducedReference,
    ReducibleData,
    Reference,
    ReferenceRun,
    Sample,
    SampleRun,
    SampleSize,
    WavelengthBins,
)
from .conversions import theta
from .corrections import normalize_by_proton_charge
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
                sam.bins.coords["wavelength"],
                sam.coords["pixel_divergence_angle"],
                sam.coords["L2"],
                ref.coords["sample_rotation"],
                ref.coords["detector_rotation"],
            ),
        ),
        c=critical_edge,
        m=mvalue,
        alpha=alpha,
    )
    return sam.bins.assign_masks(supermirror_does_not_cover=sc.isnan(R))


def evaluate_reference_at_sample_coords(
    reference: ReducedReference,
    sample: ReducibleData[SampleRun],
    detector_spatial_resolution: DetectorSpatialResolution,
    graph: CoordTransformationGraph[SampleRun],
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

    if "theta" in ref.coords:
        ref.coords.pop("theta")

    ref = ref.transform_coords(
        (
            "Q",
            "theta",
            "wavelength_resolution",
            "sample_size_resolution",
            "angular_resolution",
            "Q_resolution",
        ),
        {
            **graph,
            "wavelength_resolution": wavelength_resolution,
            "sample_size_resolution": sample_size_resolution,
            "angular_resolution": angular_resolution,
            "Q_resolution": q_resolution,
        },
        rename_dims=False,
        keep_intermediate=False,
        keep_aliases=False,
    )
    return ref


def reduce_supermirror_reference(
    reference: ReducibleData[ReferenceRun],
    proton_charge: ProtonCharge[ReferenceRun],
    wavelength_bins: WavelengthBins,
    critical_edge: CriticalEdge,
    mvalue: MValue,
    alpha: Alpha,
) -> ReducedReference:
    """
    Reduces the measurement on a supermirror to estimate the
    intensity distribution in the detector for
    an ideal sample with reflectivity :math:`R = 1`.
    """
    return reduce_reference(
        normalize_by_proton_charge(correct_by_footprint(reference), proton_charge),
        wavelength_bins,
        critical_edge,
        mvalue,
        alpha,
    )


def normalize_sample_for_supermirror_reference(
    sample: ReducibleData[SampleRun],
    proton_charge: ProtonCharge[SampleRun],
    reference: ReducedReference,
    critical_edge: CriticalEdge,
    mvalue: MValue,
    alpha: Alpha,
) -> Sample:
    """
    Applies the footprint correction and the normalization by proton charge
    to the sample measurement, and masks the events in regions of the detector
    that the supermirror reference does not cover.
    """
    sample = mask_events_where_supermirror_does_not_cover(
        sample, reference, critical_edge, mvalue, alpha
    )
    return Sample(
        normalize_by_proton_charge(correct_by_footprint(sample), proton_charge)
    )


def _mirror_detector_rows(da: sc.DataArray) -> sc.DataArray:
    """Reverse the order of the data along the scattering direction of the detector.

    The coordinates and masks are left in place.
    """
    mirrored = da
    for dim in ('blade', 'wire'):
        mirrored = sc.sort(mirrored, dim, order='descending')
    return da.assign(mirrored.data)


def reduce_direct_beam_reference(
    reference: ReducibleData[ReferenceRun],
    proton_charge: ProtonCharge[ReferenceRun],
    wavelength_bins: WavelengthBins,
) -> ReducedReference:
    """
    Reduces the measurement of the direct beam to the
    intensity distribution in the detector for
    an ideal sample with reflectivity :math:`R = 1`.

    A reflection reverses the angular profile of the beam.
    The neutrons that leave the guide at an angle :math:`\\delta` to the
    center of the beam reach the detector at :math:`\\delta` in the direct beam,
    but at :math:`-\\delta` after a reflection.
    The intensity distribution of the direct beam is therefore mirrored
    along the scattering direction of the detector.
    As in the Amor reduction software ``eos``, this is done by reversing
    the order of the detector rows.
    """
    intensity = reduce_from_events_to_lz(
        normalize_by_proton_charge(reference, proton_charge), wavelength_bins
    ).hist()
    return ReducedReference(_mirror_detector_rows(intensity))


def reduce_direct_beam_reference_with_aperture(
    reference: ReducibleData[ReferenceRun],
    proton_charge: ProtonCharge[ReferenceRun],
    wavelength_bins: WavelengthBins,
) -> ReducedReference:
    """
    Correct a direct beam restricted to the sample footprint at the reference angle.

    The reference graph supplies the incidence angle of the transmitted rays.
    Apply the footprint correction at that angle before mirroring the intensity
    and assigning the sample measurement's scattering coordinates.
    """
    return reduce_direct_beam_reference(
        correct_by_footprint(reference), proton_charge, wavelength_bins
    )


def normalize_sample_for_direct_beam_reference(
    sample: ReducibleData[SampleRun],
    proton_charge: ProtonCharge[SampleRun],
) -> Sample:
    """
    Applies the footprint correction and the normalization by proton charge
    to the sample measurement.

    For an unrestricted direct beam, only the sample needs this correction.
    For a reference measured through a footprint aperture, the reference is
    corrected separately at its own incidence angles.
    """
    return Sample(
        normalize_by_proton_charge(correct_by_footprint(sample), proton_charge)
    )


def sample_size_for_direct_beam(
    sample_size: SampleSize[SampleRun],
) -> SampleSize[ReferenceRun]:
    """There is no sample in the direct beam, the size of the sample
    in the sample run is used where the reference is evaluated at the
    coordinates of the sample run."""
    return SampleSize[ReferenceRun](sample_size)


providers = (evaluate_reference_at_sample_coords,)
"""Providers that do not depend on the kind of reference measurement."""

supermirror_providers = (
    reduce_supermirror_reference,
    normalize_sample_for_supermirror_reference,
)
"""Providers for normalizing by a reference measured on a supermirror."""

direct_beam_providers = (
    reduce_direct_beam_reference,
    normalize_sample_for_direct_beam_reference,
    sample_size_for_direct_beam,
)
"""Providers for normalizing by a measurement of the direct beam."""
