# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Masking functions for the loki workflow.
"""
from typing import Optional

import numpy as np
import sciline
import scipp as sc

from .types import (
    MaskedData,
    MaskedDetectorIDs,
    PixelMask,
    PixelMaskFilename,
    RawData,
    SampleRun,
    ScatteringRunType,
)


def to_pixel_mask(data: RawData[SampleRun], masked: MaskedDetectorIDs) -> PixelMask:
    """Convert a list of masked detector IDs to a pixel mask.

    Parameters
    ----------
    data:
        Raw data, defining the detector IDs.
    masked:
        The masked detector IDs.
    """
    ids = data.coords['detector_number']
    mask = sc.zeros(sizes=ids.sizes, dtype='bool')
    mask.values[np.isin(ids.values, masked.values)] = True
    return PixelMask(mask)


def apply_pixel_masks(
    data: RawData[ScatteringRunType],
    masks: Optional[sciline.Series[PixelMaskFilename, PixelMask]],
) -> MaskedData[ScatteringRunType]:
    """Apply pixel-specific masks to raw data.
    The masks are based on detector IDs stored in XML files.

    Parameters
    ----------
    data:
        Raw data with configured component positions.
    masks:
        A series of masks.
    """
    if masks is not None:
        data = data.copy(deep=False)
        for name, mask in masks.items():
            data.masks[name] = mask
    return MaskedData[ScatteringRunType](data)


providers = (apply_pixel_masks, to_pixel_mask)
