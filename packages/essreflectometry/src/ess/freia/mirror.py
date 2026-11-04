# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Geometry for FREIA's planar inversion mirror."""

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
from .types import MirrorGeometry, MirrorNormal, MirrorPosition


def no_mirror() -> MirrorGeometry[RunType] | None:
    """Use the ordinary geometry when inversion is disabled."""
    return None


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


def mirror_geometry(
    position: MirrorPosition[RunType],
    normal: MirrorNormal[RunType],
) -> MirrorGeometry[RunType] | None:
    """Validate the mirror position and normalize its surface normal."""
    for name, vector in [('MirrorPosition', position), ('MirrorNormal', normal)]:
        if vector.ndim != 0 or vector.dtype != sc.DType.vector3:
            raise ValueError(f'{name} must be a scalar vector.')
        if not sc.isfinite(vector).value:
            raise ValueError(f'{name} must be finite.')
    length = sc.norm(normal)
    if not sc.isfinite(length).value or length.value == 0:
        raise ValueError('MirrorNormal must have a finite, nonzero norm.')
    return MirrorGeometry(position=position, normal=normal / length)


def reflected_upstream_position(
    upstream: sc.Variable,
    downstream: sc.Variable,
    mirror: MirrorGeometry | None = None,
) -> sc.Variable:
    r"""Return the upstream position reflected across the mirror plane.

    Let :math:`p` be ``upstream``, :math:`m` be ``mirror.position``, and :math:`n`
    be the unit normal ``mirror.normal``.
    The returned position :math:`p'` is

    .. math::

        p' = p - 2n\,[n \cdot (p-m)].

    The straight segment from :math:`p'` to ``downstream`` has the same
    length as the path through the mirror and points along its final segment.
    The returned point is used for geometry calculations; the physical
    component remains in place. With no mirror, return ``upstream``
    unchanged. The result retains the upstream position's length unit.

    Raises
    ------
    ValueError
        If the physical upstream and downstream points are on opposite sides
        of the mirror plane, so no reflected path connects them.
    """
    if mirror is None:
        return upstream
    position = mirror.position.to(unit=upstream.unit)
    upstream_distance = sc.dot(upstream - position, mirror.normal)
    downstream_distance = sc.dot(
        downstream.to(unit=upstream.unit) - position, mirror.normal
    )
    if (upstream_distance * downstream_distance).value < 0:
        raise ValueError(
            'A reflected path requires its upstream and downstream positions '
            'to be on the same side of the mirror plane. Check the mirror pose '
            'and component positions.'
        )
    return upstream - 2 * mirror.normal * upstream_distance


def detector_ltotal_with_mirror(
    detector: EmptyDetector[RunType],
    source_position: Position[snx.NXsource, RunType],
    sample_position: Position[snx.NXsample, RunType],
    mirror: MirrorGeometry[RunType] | None,
) -> DetectorLtotal[RunType]:
    """Compute the source-to-detector flight length through the mirror and sample.

    As in the ordinary workflow, this approximates the upstream guide by a
    straight ray. Reflecting the source position across the mirror plane gives
    the length of the path from source through mirror to sample. Chopper and
    upstream-monitor wavelength reconstruction use the physical source position.
    """
    reflected_source = reflected_upstream_position(
        source_position, sample_position, mirror
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
    mirror_geometry,
    detector_ltotal_with_mirror,
)
