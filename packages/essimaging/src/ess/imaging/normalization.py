# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Contains the providers for normalization.
"""

import scipp as sc

from .types import (
    BackgroundSubtractedDetector,
    CorrectedDetector,
    DarkBackgroundRun,
    IntensityDetector,
    NormalizedDetector,
    OpenBeamRun,
    ProtonCharge,
    RunType,
    SampleRun,
)


def subtract_background_sample(
    data: NormalizedDetector[SampleRun],
    background: NormalizedDetector[DarkBackgroundRun],
) -> BackgroundSubtractedDetector[SampleRun]:
    """
    Subtract background from proton-charge normalized sample data.

    Parameters
    ----------
    data:
        Sample data to process (normalized to proton charge).
    background:
        Background (dark frame) data to subtract (normalized to proton charge).
    """
    return BackgroundSubtractedDetector[SampleRun](data - background)


def subtract_background_openbeam(
    data: NormalizedDetector[OpenBeamRun],
    background: NormalizedDetector[DarkBackgroundRun],
) -> BackgroundSubtractedDetector[OpenBeamRun]:
    """
    Subtract background from proton-charge normalized open beam data.

    Parameters
    ----------
    data:
        Open beam data to process (normalized to proton charge).
    background:
        Background (dark frame) data to subtract (normalized to proton charge).
    """
    return BackgroundSubtractedDetector[OpenBeamRun](data - background)


def sample_over_openbeam(
    sample: BackgroundSubtractedDetector[SampleRun],
    open_beam: BackgroundSubtractedDetector[OpenBeamRun],
) -> IntensityDetector:
    """
    Divide background-subtracted sample data by background-subtracted open beam data,
    to obtain final normalized image.

    Parameters
    ----------
    sample:
        Sample data to process (background subtracted).
    open_beam:
        Open beam data to divide by (background subtracted).
    """
    return IntensityDetector(sample / open_beam)


providers = (
    subtract_background_sample,
    subtract_background_openbeam,
    sample_over_openbeam,
)
