# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Masking functions for the loki workflow.
"""

from ess.reduce.masking import read_pixel_masks, to_pixel_masks

from .types import (
    CorrectedDetector,
    DetectorIDs,
    DetectorMasks,
    EmptyDetector,
    LoadedPixelMasks,
    Numerator,
    RunType,
    SampleRun,
    WavelengthDetector,
)


def get_detector_ids_from_detector(data: EmptyDetector[SampleRun]) -> DetectorIDs:
    """Extract detector IDs from a detector."""
    return DetectorIDs(
        data.coords[
            'detector_number' if 'detector_number' in data.coords else 'detector_id'
        ]
    )


def to_detector_mask(ids: DetectorIDs, masks: LoadedPixelMasks) -> DetectorMasks:
    """Create detector masks from the masks read from files.

    Parameters
    ----------
    ids:
        Array of detector IDs
    masks:
        Masks read from files.
    """
    return DetectorMasks(to_pixel_masks(masks, ids))


def apply_pixel_masks(
    data: WavelengthDetector[RunType],
    masks: DetectorMasks,
) -> CorrectedDetector[RunType, Numerator]:
    """Apply pixel-specific masks to raw data.

    Parameters
    ----------
    data:
        Raw data with configured component positions.
    masks:
        A series of masks.
    """
    return CorrectedDetector[RunType, Numerator](data.assign_masks(masks))


providers = (
    get_detector_ids_from_detector,
    read_pixel_masks,
    to_detector_mask,
    apply_pixel_masks,
)
