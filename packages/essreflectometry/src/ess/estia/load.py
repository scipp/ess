# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import scipp as sc
from ess.reduce.nexus.types import NeXusTransformation
from scippnexus import NXdetector, NXsample

from ..reflectometry.types import (
    DetectorRotation,
    RawSampleRotation,
    RunType,
)


def _orientation_axis(
    transform: sc.Variable | sc.DataArray, axis: tuple[int, int, int]
) -> sc.Variable:
    if isinstance(transform, sc.DataArray) or transform.ndim != 0:
        raise ValueError(
            'ESTIA requires a static orientation. Select a time interval or '
            'provide a TransformationTimeFilter.'
        )
    # Transform a direction, cancelling the translation in the affine transform.
    origin = sc.vector([0, 0, 0], unit=transform.unit)
    return transform * sc.vector(axis, unit=transform.unit) - transform * origin


def load_sample_rotation(
    transform: NeXusTransformation[NXsample, RunType],
) -> RawSampleRotation[RunType]:
    """Compute the sample surface's horizontal angle from its full NeXus chain.

    The local surface normal is +x. Its projection onto the laboratory x-z
    plane defines the surface angle used by ESTIA's horizontal reduction.
    This includes rotations of the hexapod as well as the sample stage.
    """
    normal = _orientation_axis(transform.value, (1, 0, 0))
    return sc.atan2(y=-normal.fields.z, x=normal.fields.x).to(unit='deg')


def load_detector_rotation(
    transform: NeXusTransformation[NXdetector, RunType],
) -> DetectorRotation[RunType]:
    """Compute the horizontal angle of the detector's local +z direction.

    Use the same composed, time-filtered transform as the detector positions.
    """
    direction = _orientation_axis(transform.value, (0, 0, 1))
    return sc.atan2(y=direction.fields.x, x=direction.fields.z).to(unit='deg')


providers = (load_sample_rotation, load_detector_rotation)
