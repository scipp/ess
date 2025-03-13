# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry.conversions import reflectometry_q
from ..reflectometry.supermirror import (
    Alpha,
    CriticalEdge,
    MValue,
    supermirror_reflectivity,
)
from ..reflectometry.types import (
    CoordTransformationGraph,
    DetectorSpatialResolution,
    ReducedReference,
    ReducibleData,
    Reference,
    Sample,
    SampleRun,
)
from .conversions import theta
from .resolution import (
    angular_resolution,
    q_resolution,
    sample_size_resolution,
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
                sam.coords["divergence_angle"],
                ref.coords["sample_rotation"],
            ),
        ),
        c=critical_edge,
        m=mvalue,
        alpha=alpha,
    )
    return sam.bins.assign_masks(supermirror_does_not_cover=sc.isnan(R))


def evaluate_reference(
    reference: ReducedReference,
    sample: ReducibleData[SampleRun],
    graph: CoordTransformationGraph,
    detector_spatial_resolution: DetectorSpatialResolution[SampleRun],
) -> Reference:
    """
    Adds a :math:`Q` and :math:`Q`-resolution coordinate to each bin of the ideal
    intensity distribution. The coordinates are computed as if the data came from
    the sample measurement, that is, they use the ``sample_rotation``
    and ``detector_rotation`` parameters from the sample measurement.
    """
    ref = reference.copy()
    ref.coords["sample_rotation"] = sample.coords['sample_rotation']
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
            # The wavelength resolution of Estia is not implemented yet
            # when it is this will be replaced by the correct value.
            "wavelength_resolution": lambda: sc.scalar(0.05),
            "sample_size_resolution": sample_size_resolution,
            "angular_resolution": angular_resolution,
            "Q_resolution": q_resolution,
        },
        rename_dims=False,
        keep_intermediate=False,
        keep_aliases=False,
    )
    return ref


providers = (
    evaluate_reference,
    mask_events_where_supermirror_does_not_cover,
)
