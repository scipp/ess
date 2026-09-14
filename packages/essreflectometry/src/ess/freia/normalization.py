# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import scipp as sc
from ess.reduce.nexus.types import GravityVector

from ..reflectometry.conversions import reflectometry_q
from ..reflectometry.types import (
    QBins,
    ReducibleData,
    Reference,
    ReferenceRun,
    ReflectivityOverQ,
    Sample,
    SampleRun,
)
from .conversions import theta
from .types import SampleSurfaceNormal


def evaluate_direct_beam(
    direct_beam: ReducibleData[ReferenceRun],
    sample_surface_normal: SampleSurfaceNormal[SampleRun],
    gravity: GravityVector,
) -> Reference:
    """Map the direct-beam ROI to Q after reflection at the sample surface.

    A direct ray follows the incident direction. Specular reflection reverses
    its normal component, so its exit angle is the negative of its angle above
    the sample plane. Using the sample run's normal also supports a changed
    sample orientation. No footprint or supermirror correction is applied.
    """
    angle = -theta(
        incident_beam=direct_beam.coords['incident_beam'],
        scattered_beam=direct_beam.coords['position']
        - direct_beam.coords['sample_position'],
        wavelength=direct_beam.bins.coords['wavelength'],
        gravity=gravity,
        sample_surface_normal=sample_surface_normal,
    )
    reference = direct_beam.bins.assign_coords(
        theta=angle, Q=reflectometry_q(direct_beam.bins.coords['wavelength'], angle)
    )
    return Reference(
        reference.bins.assign_masks(non_incident=angle <= sc.scalar(0.0, unit='rad'))
    )


def reduce_sample_over_q(
    sample: Sample,
    reference: Reference,
    qbins: QBins,
) -> ReflectivityOverQ:
    """Divide ROI intensities on a common Q grid, propagating both variances.

    Histogram before division so changing Q bin widths does not rescale R.
    Empty, masked, or nonfinite direct-beam bins provide no normalization.
    """
    numerator = sample.hist(Q=qbins, dim=sample.dims)
    denominator = reference.hist(Q=qbins, dim=reference.dims)
    valid = sc.isfinite(denominator.data) & (
        denominator.data > sc.scalar(0.0, unit=denominator.unit)
    )
    # Keep unusable bins explicit instead of producing infinities at empty bins.
    norm = sc.where(
        valid, denominator.data, sc.scalar(float('nan'), unit=denominator.unit)
    )
    return ReflectivityOverQ((numerator / norm).assign_masks(direct_beam=~valid))


providers = (evaluate_direct_beam, reduce_sample_over_q)
