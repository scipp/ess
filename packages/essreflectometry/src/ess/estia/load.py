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
    transform: sc.Variable | sc.DataArray, axis: tuple[float, float, float]
) -> sc.Variable:
    # Transform a direction, cancelling the translation in the affine transform.
    origin = sc.vector([0, 0, 0], unit=transform.unit)
    return transform * sc.vector(axis, unit=transform.unit) - transform * origin


def load_sample_rotation(
    transform: NeXusTransformation[NXsample, RunType],
) -> RawSampleRotation[RunType]:
    """Compute the sample mount angle from its NeXus transformations.

    Project the mount's transformed local +x axis onto the laboratory x-z plane
    and measure its angle from +x towards -z. The sample may be tilted within
    its mount. ``SampleRotationOffset`` calibrates the resulting angle.
    """
    normal = _orientation_axis(transform.value, (1, 0, 0))
    return sc.atan2(y=-normal.fields.z, x=normal.fields.x).to(unit='deg')


def load_detector_rotation(
    transform: NeXusTransformation[NXdetector, RunType],
) -> DetectorRotation[RunType]:
    """Compute the detector reference angle from its NeXus transformations.

    Measure the transformed local +z direction from laboratory +z towards +x
    in the x-z plane. The reduction assumes this direction points from the
    sample to the detector centre, where the central reflected beam is aligned.
    Use the same transform as the detector pixel positions.
    """
    direction = _orientation_axis(transform.value, (0, 0, 1))
    return sc.atan2(y=direction.fields.x, x=direction.fields.z).to(unit='deg')


providers = (load_sample_rotation, load_detector_rotation)
