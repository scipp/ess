# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import numpy as np
import sciline
import scipp as sc

from ..types import (
    MaskedData,
    MaskedDetectorIDs,
    PixelMask,
    RawData,
    SampleRun,
    ScatteringRunType,
)
from .components import RawDataWithComponentUserOffsets


def to_pixel_mask(data: RawData[SampleRun], masked: MaskedDetectorIDs) -> PixelMask:
    """Convert a list of masked detector IDs to a pixel mask.

    Parameters
    ----------
    data:
        Raw data, defining the detector IDs.
    masked:
        The masked detector IDs.
    """
    ids = data.coords['detector_id']
    mask = sc.zeros(sizes=ids.sizes, dtype='bool')
    mask.values[np.isin(ids.values, masked.values)] = True
    return PixelMask(mask)


def apply_pixel_masks(
    data: RawDataWithComponentUserOffsets[ScatteringRunType],
    masks: sciline.Series[str, PixelMask],
) -> MaskedData[ScatteringRunType]:
    """Apply pixel-specific masks to raw data.

    This depends on the configured raw data (which has been configured with component
    positions) since in principle we might apply pixel masks based on the component
    positions. Currently the only masks are based on detector IDs.

    Parameters
    ----------
    data:
        Raw data with configured component positions.
    masks:
        A series of masks.
    """
    data = data.copy(deep=False)
    for name, mask in masks.items():
        data.masks[name] = mask
    return MaskedData[ScatteringRunType](data)


providers = (
    to_pixel_mask,
    apply_pixel_masks,
)
