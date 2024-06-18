# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Masking functions for the loki workflow.
"""

import numpy as np
import scipp as sc

from .types import (
    DetectorIDs,
    DetectorMasks,
    MaskedData,
    MaskedDetectorIDs,
    PixelMaskFilename,
    SampleRun,
    ScatteringRunType,
    TofData,
)


def get_detector_ids_from_sample_run(data: TofData[SampleRun]) -> DetectorIDs:
    """Extract detector IDs from sample run."""
    return DetectorIDs(
        data.coords[
            'detector_number' if 'detector_number' in data.coords else 'detector_id'
        ]
    )


def to_detector_mask(
    ids: DetectorIDs,
    name: PixelMaskFilename,
    masked_ids: MaskedDetectorIDs,
) -> DetectorMasks:
    """Create a detector mask from a list of masked detector IDs.
    The masks are based on detector IDs stored in XML files.

    Parameters
    ----------
    data:
        Raw data with configured component positions.
    masked_ids:
        Detector IDs to mask.
    """
    mask = sc.zeros(sizes=ids.sizes, dtype='bool')
    mask.values[np.isin(ids.values, masked_ids.values)] = True
    return DetectorMasks({name: mask})


def apply_pixel_masks(
    data: TofData[ScatteringRunType],
    masks: DetectorMasks,
) -> MaskedData[ScatteringRunType]:
    """Apply pixel-specific masks to raw data.

    Parameters
    ----------
    data:
        Raw data with configured component positions.
    masks:
        A series of masks.
    """
    return MaskedData[ScatteringRunType](data.assign_masks(masks))


providers = (
    get_detector_ids_from_sample_run,
    to_detector_mask,
    apply_pixel_masks,
)
