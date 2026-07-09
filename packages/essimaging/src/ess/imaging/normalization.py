# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Contains the providers for normalization.
"""

from ess.reduce.uncertainty import broadcast_uncertainties

from .types import (
    BackgroundSubtractedDetector,
    CorrectedDetector,
    DarkBackgroundRun,
    FluxNormalizedDetector,
    MeanDarkFrame,
    NormalizedImage,
    OpenBeamRun,
    SampleRun,
    UncertaintyBroadcastMode,
)


def average_dark_frames(
    dark_frames: CorrectedDetector[DarkBackgroundRun],
) -> MeanDarkFrame:
    """
    Average the dark frames (background runs) over time to obtain a mean dark frame.

    Parameters
    ----------
    dark_frames:
        Dark frames to average.
    """
    return MeanDarkFrame(dark_frames.mean('time'))


def subtract_background_sample(
    data: CorrectedDetector[SampleRun],
    background: MeanDarkFrame,
    uncertainties: UncertaintyBroadcastMode,
) -> BackgroundSubtractedDetector[SampleRun]:
    """
    Subtract background from sample data.

    Parameters
    ----------
    data:
        Sample data to process.
    background:
        Background (dark frame) data to subtract.
    uncertainties:
        Mode to use when broadcasting uncertainties from background to data (data
        typically has multiple frames, while background usually has only one frame).
    """
    return BackgroundSubtractedDetector[SampleRun](
        data - broadcast_uncertainties(background, prototype=data, mode=uncertainties)
    )


def subtract_background_openbeam(
    data: CorrectedDetector[OpenBeamRun],
    background: MeanDarkFrame,
    uncertainties: UncertaintyBroadcastMode,
) -> BackgroundSubtractedDetector[OpenBeamRun]:
    """
    Subtract background from open beam data.

    Parameters
    ----------
    data:
        Open beam data to process.
    background:
        Background (dark frame) data to subtract.
    uncertainties:
        Mode to use when broadcasting uncertainties from background to data (data
        typically has multiple frames, while background usually has only one frame).
    """
    return BackgroundSubtractedDetector[OpenBeamRun](
        data - broadcast_uncertainties(background, prototype=data, mode=uncertainties)
    )


def sample_over_openbeam(
    sample: FluxNormalizedDetector[SampleRun],
    open_beam: FluxNormalizedDetector[OpenBeamRun],
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
        / broadcast_uncertainties(
            open_beam.mean('time'), prototype=sample, mode=uncertainties
        )
    )


providers = (
    average_dark_frames,
    subtract_background_sample,
    subtract_background_openbeam,
    sample_over_openbeam,
)
