# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Geometry for FREIA's planar inversion mirror."""

import scipp as sc
import scippnexus as snx
from ess.reduce.nexus.types import (
    EmptyDetector,
    NeutronSourcePosition,
    NeXusTransformation,
    Position,
)
from ess.reduce.unwrap.types import DetectorLtotal

from ..reflectometry.types import RunType
from .types import MirrorGeometry


def no_mirror() -> MirrorGeometry[RunType] | None:
    """Use the ordinary geometry when inversion is disabled."""
    return None


def mirror_geometry(
    position: Position[snx.NXmirror, RunType],
    transformation: NeXusTransformation[snx.NXmirror, RunType],
) -> MirrorGeometry[RunType] | None:
    """Build the mirror plane from its shared NeXus position and transformation.

    Rotate local -z, the NXmirror reference-plane normal, to the lab frame.
    Subtract the transformed origin to cancel translation, independently of
    any override of the component position.
    """
    transform = transformation.value
    origin = transform * sc.vector([0.0, 0.0, 0.0], unit='m')
    tip = transform * sc.vector([0.0, 0.0, -1.0], unit='m')
    return MirrorGeometry(position=position, normal=tip - origin)


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
    source_position: NeutronSourcePosition[RunType],
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
    mirror_geometry,
    detector_ltotal_with_mirror,
)
