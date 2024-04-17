# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Masking functions for the loki workflow.
"""
import numpy as np
import scipp as sc

from .types import (
    DetectorMasks,
    MaskedData,
    MaskedDetectorIDs,
    PixelMaskFilename,
    ScatteringRunType,
    TofData,
)


def to_detector_mask(
    data: TofData[ScatteringRunType],
    name: PixelMaskFilename,
    masked_ids: MaskedDetectorIDs,
) -> DetectorMasks[ScatteringRunType]:
    """Create a detector mask from a list of masked detector IDs.
    The masks are based on detector IDs stored in XML files.

    Parameters
    ----------
    data:
        Raw data with configured component positions.
    masked_ids:
        Detector IDs to mask.
    """
    ids = data.coords[
        'detector_number' if 'detector_number' in data.coords else 'detector_id'
    ]
    mask = sc.zeros(sizes=ids.sizes, dtype='bool')
    mask.values[np.isin(ids.values, masked_ids.values)] = True
    return DetectorMasks[ScatteringRunType]({name: mask})


def apply_pixel_masks(
    data: TofData[ScatteringRunType],
    masks: DetectorMasks[ScatteringRunType],
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
    to_detector_mask,
    apply_pixel_masks,
)
