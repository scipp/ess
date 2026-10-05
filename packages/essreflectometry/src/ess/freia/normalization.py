# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry.conversions import reflectometry_q
from ..reflectometry.types import (
    QBins,
    ReducibleData,
    Reference,
    ReferenceRun,
    ReflectivityOverQ,
    Sample,
)
from .conversions import theta


def evaluate_direct_beam(
    direct_beam: ReducibleData[ReferenceRun],
) -> Reference:
    """Compute reference Q for the direct beam reflected in the sample plane."""
    normal = direct_beam.coords['sample_surface_normal']
    normal = normal / sc.norm(normal)
    outgoing = direct_beam.bins.coords['outgoing_direction']
    # The reflected direction is the direction the direct beam
    # would have gone if it had been reflected.
    # It is the direct beam reflected by the sample surface.
    reflected = outgoing - 2 * sc.dot(outgoing, normal) * normal
    reflection_angle = theta(
        outgoing_direction=reflected,
        sample_surface_normal=normal,
    )
    wavelength = direct_beam.bins.coords['wavelength']
    return Reference(
        direct_beam.bins.assign_coords(Q=reflectometry_q(wavelength, reflection_angle))
    )


def reduce_sample_over_q(
    sample: Sample,
    reference: Reference,
    qbins: QBins,
) -> ReflectivityOverQ:
    """Divide ROI intensities on a common Q grid, propagating both variances."""
    numerator = sample.hist(Q=qbins, dim=sample.dims)
    denominator = reference.hist(Q=qbins, dim=reference.dims)
    valid = sc.isfinite(denominator.data) & (
        denominator.data > sc.scalar(0.0, unit=denominator.unit)
    )
    return ReflectivityOverQ(
        (numerator / denominator.data).assign_masks(direct_beam=~valid)
    )


providers = (evaluate_direct_beam, reduce_sample_over_q)
