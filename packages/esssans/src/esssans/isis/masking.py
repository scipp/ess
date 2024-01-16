# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import sciline
import scipp as sc

from ..types import MaskedData, RawData, RunType
from .mantid_io import PixelMask


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
        if not sc.identical(mask.coords['spectrum'], data.coords['spectrum']):
            raise ValueError(f"Mask {name} has different spectrum numbers than data")
        data.masks[name] = mask.data
    return MaskedData[RunType](data)
