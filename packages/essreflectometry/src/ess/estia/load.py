# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from scippnexus import NXdetector, NXsample

from ess.reduce.nexus.types import NeXusComponent

from ..reflectometry.types import (
    DetectorRotation,
    RawSampleRotation,
    RunType,
)


def load_sample_rotation(
    sample: NeXusComponent[NXsample, RunType],
) -> RawSampleRotation[RunType]:
    return sample['sample_rotation'][0].data


def load_detector_rotation(
    detector: NeXusComponent[NXdetector, RunType],
) -> DetectorRotation[RunType]:
    return detector['transformations']['detector_rotation'].value[0].data


providers = (load_sample_rotation, load_detector_rotation)
