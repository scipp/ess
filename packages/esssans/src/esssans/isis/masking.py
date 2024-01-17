# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import sciline

from ..types import MaskedData, RawData, RunType
from .mantidio import PixelMask


def apply_pixel_masks(
    data: RawData[RunType], masks: sciline.Series[str, PixelMask]
) -> MaskedData[RunType]:
    """Apply pixel-specific masks to raw data.

    Parameters
    ----------
    data:
        Raw data.
    masks:
        A series of masks.
    """
    data = data.copy(deep=False)
    for name, mask in masks.items():
        data.masks[name] = mask
    return MaskedData[RunType](data)
