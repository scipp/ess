# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Contains the providers for normalization.
"""

from ess.reduce.uncertainty import broadcast_uncertainties

from .types import (
    BackgroundSubtractedDetector,
    DarkBackgroundRun,
    FluxNormalizedDetector,
    NormalizedImage,
    OpenBeamRun,
    SampleRun,
    UncertaintyBroadcastMode,
)


def subtract_background_sample(
    data: FluxNormalizedDetector[SampleRun],
    background: FluxNormalizedDetector[DarkBackgroundRun],
    uncertainties: UncertaintyBroadcastMode,
) -> BackgroundSubtractedDetector[SampleRun]:
    """
    Subtract background from proton-charge normalized sample data.

    Parameters
    ----------
    data:
        Sample data to process (normalized to proton charge).
    background:
        Background (dark frame) data to subtract (normalized to proton charge).
    uncertainties:
        Mode to use when broadcasting uncertainties from background to data (data
        typically has multiple frames, while background usually has only one frame).
    """
    return BackgroundSubtractedDetector[SampleRun](
        data - broadcast_uncertainties(background, prototype=data, mode=uncertainties)
    )


def subtract_background_openbeam(
    data: FluxNormalizedDetector[OpenBeamRun],
    background: FluxNormalizedDetector[DarkBackgroundRun],
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
    uncertainties: UncertaintyBroadcastMode,
) -> NormalizedImage:
    """
    Divide background-subtracted sample data by background-subtracted open beam data,
    to obtain final normalized image.

    Parameters
    ----------
    sample:
        Sample data to process (background subtracted).
    open_beam:
        Open beam data to divide by (background subtracted).
    uncertainties:
        Mode to use when broadcasting uncertainties from open beam to sample (sample
        typically has multiple frames, while open beam usually has only one frame).
    """
    return NormalizedImage(
        sample
        / broadcast_uncertainties(open_beam, prototype=sample, mode=uncertainties)
    )


providers = (
    subtract_background_sample,
    subtract_background_openbeam,
    sample_over_openbeam,
)
