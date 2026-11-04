# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Geometry for FREIA's planar inversion mirror."""

import numpy as np
import scipp as sc
import scippnexus as snx
from ess.reduce import nexus
from ess.reduce.nexus.types import (
    EmptyDetector,
    NeXusComponentLocationSpec,
    NeXusFileSpec,
    NeXusName,
    NeXusTransformation,
    Position,
    TimeInterval,
    TransformationTimeFilter,
)
from ess.reduce.nexus.workflow import reject_time_dependent_transform, to_transformation
from ess.reduce.unwrap.types import DetectorLtotal

from ..reflectometry.types import RunType
from .types import IncidentBeamTransform, MirrorNormal, MirrorPosition

# Default for direct calls to coordinate-graph providers; never modified in place.
DIRECT_BEAM = sc.spatial.translation(value=[0.0, 0.0, 0.0], unit='m')


def direct_beam_transform() -> IncidentBeamTransform[RunType]:
    """Leave upstream coordinates unchanged when inversion is disabled."""
    return IncidentBeamTransform[RunType](DIRECT_BEAM.copy())


def mirror_time_filter() -> TransformationTimeFilter[snx.NXmirror, RunType]:
    """Require one mirror pose, unless the caller explicitly selects a pose."""
    return reject_time_dependent_transform


def load_mirror_transformation(
    file_spec: NeXusFileSpec[RunType],
    name: NeXusName[snx.NXmirror],
    interval: TimeInterval[RunType],
    time_filter: TransformationTimeFilter[
        snx.NXmirror, RunType
    ] = reject_time_dependent_transform,
) -> NeXusTransformation[snx.NXmirror, RunType]:
    """Load the mirror's transformation for the selected time interval.

    The component name is a provisional convention, configurable through
    ``NeXusName[NXmirror]``. No pose is inferred if the component is missing.
    """
    location = NeXusComponentLocationSpec[snx.NXmirror, RunType](
        filename=file_spec.value, component_name=name
    )
    try:
        component = nexus.load_component(location, nx_class=snx.NXmirror)
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"Cannot load inversion mirror {name!r}. Set NeXusName[NXmirror] "
            "to its component name, or supply MirrorPosition and MirrorNormal "
            "for this run."
        ) from error
    if 'depends_on' not in component:
        raise ValueError("The inversion NXmirror requires a depends_on chain.")
    return to_transformation(component['depends_on'], interval, time_filter)


def mirror_position(
    transformation: NeXusTransformation[snx.NXmirror, RunType],
) -> MirrorPosition[RunType]:
    """Transform the NXmirror origin to a point on the reflecting plane."""
    transform = transformation.value
    if isinstance(transform, sc.DataArray):
        raise ValueError('The inversion mirror must have a static pose for the run.')
    return MirrorPosition[RunType](transform * sc.vector([0.0, 0.0, 0.0], unit='m'))


def mirror_normal(
    transformation: NeXusTransformation[snx.NXmirror, RunType],
) -> MirrorNormal[RunType]:
    """Rotate local -z, the NXmirror reference-plane normal, to the lab frame."""
    origin = mirror_position(transformation)
    tip = transformation.value * sc.vector([0.0, 0.0, -1.0], unit='m')
    normal = tip - origin
    return MirrorNormal[RunType](normal / sc.norm(normal))


def mirror_reflection_transform(
    position: MirrorPosition[RunType],
    normal: MirrorNormal[RunType],
) -> IncidentBeamTransform[RunType]:
    r"""Build a transform that reflects positions across the mirror plane.

    For a point :math:`m` on the plane and its unit normal :math:`n`, the
    transform maps a position :math:`p` to

    .. math::

        p' = p - 2n\,[n \cdot (p-m)].

    The magnitude and sign of ``normal`` do not affect the transform.
    """
    for name, vector in [('MirrorPosition', position), ('MirrorNormal', normal)]:
        if vector.ndim != 0 or vector.dtype != sc.DType.vector3:
            raise ValueError(f'{name} must be a scalar vector.')
        if not np.isfinite(vector.value).all():
            raise ValueError(f'{name} must be finite.')
    n = normal.value
    scale = np.max(np.abs(n))
    if scale == 0:
        raise ValueError('MirrorNormal must be nonzero.')
    n = n / scale
    n = n / np.linalg.norm(n)
    matrix = np.eye(4)
    matrix[:3, :3] -= 2 * np.outer(n, n)
    p = position.to(unit='m').value
    matrix[:3, 3] = p - matrix[:3, :3] @ p
    return IncidentBeamTransform[RunType](
        sc.spatial.affine_transform(value=matrix, unit='m')
    )


def transform_position(transform: sc.Variable, position: sc.Variable) -> sc.Variable:
    """Apply the beam-path transform, preserving the input length unit."""
    return (transform * position.to(unit=transform.unit)).to(unit=position.unit)


def reflected_upstream_position(
    transform: sc.Variable, upstream: sc.Variable, downstream: sc.Variable
) -> sc.Variable:
    """Return the upstream position reflected across the mirror plane.

    The straight segment from this position to ``downstream`` has the same
    length as the path through the mirror and points along its final segment.
    The returned point is used for geometry calculations; the physical
    component remains in place. With ``DIRECT_BEAM``, return ``upstream``
    unchanged. The result retains the upstream position's length unit.

    Raises
    ------
    ValueError
        If the physical upstream and downstream points are on opposite sides
        of the mirror plane, so no reflected path connects them.
    """
    if sc.identical(transform, DIRECT_BEAM):
        return upstream
    reflected = transform_position(transform, upstream)
    midpoint = (0.5 * (upstream + reflected)).to(unit=downstream.unit)
    normal = (upstream - reflected).to(unit=downstream.unit)
    if sc.dot(downstream - midpoint, normal).value < 0:
        raise ValueError(
            'A reflected path requires its upstream and downstream positions '
            'to be on the same side of the mirror plane. Check the mirror pose '
            'and component positions.'
        )
    return reflected


def detector_ltotal_with_mirror(
    detector: EmptyDetector[RunType],
    source_position: Position[snx.NXsource, RunType],
    sample_position: Position[snx.NXsample, RunType],
    transform: IncidentBeamTransform[RunType],
) -> DetectorLtotal[RunType]:
    """Compute the source-to-detector flight length through the mirror and sample.

    As in the ordinary workflow, this approximates the upstream guide by a
    straight ray. Reflecting the source position across the mirror plane gives
    the length of the path from source through mirror to sample. Chopper and
    upstream-monitor wavelength reconstruction use the physical source position.
    """
    reflected_source = reflected_upstream_position(
        transform, source_position, sample_position
    )
    sample = sample_position.to(unit=reflected_source.unit)
    pixels = detector.coords['position'].to(unit=reflected_source.unit)
    return DetectorLtotal[RunType](
        sc.norm(sample - reflected_source) + sc.norm(pixels - sample)
    )


providers = (
    mirror_time_filter,
    load_mirror_transformation,
    mirror_position,
    mirror_normal,
    mirror_reflection_transform,
    detector_ltotal_with_mirror,
)
