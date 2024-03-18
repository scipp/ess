# # SPDX-License-Identifier: BSD-3-Clause
# # Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# """
# Masking functions for the loki workflow.
# """
# from typing import Optional

# import numpy as np
# import sciline
# import scipp as sc

# from .types import (
#     # LogicalDimsData,
#     MaskedData,
#     MaskedDetectorIDs,
#     PixelMask,
#     PixelMaskFilename,
#     SampleRun,
#     ScatteringRunType,
#     TofData,
# )


# def to_pixel_mask(data: TofData[SampleRun], masked: MaskedDetectorIDs) -> PixelMask:
#     """Convert a list of masked detector IDs to a pixel mask.

#     Parameters
#     ----------
#     data:
#         Raw data, defining the detector IDs.
#     masked:
#         The masked detector IDs.
#     """
#     ids = data.coords['detector_number']
#     mask = sc.zeros(sizes=ids.sizes, dtype='bool')
#     mask.values[np.isin(ids.values, masked.values)] = True
#     return PixelMask(mask)


# def apply_pixel_masks(
#     data: TofData[ScatteringRunType],
#     masks: Optional[sciline.Series[PixelMaskFilename, PixelMask]],
# ) -> MaskedData[ScatteringRunType]:
#     """Apply pixel-specific masks to raw data.
#     The masks are based on detector IDs stored in XML files.

#     Parameters
#     ----------
#     data:
#         Raw data with configured component positions.
#     masks:
#         A series of masks.
#     """
#     if masks is not None:
#         data = data.copy(deep=False)
#         for name, mask in masks.items():
#             data.masks[name] = mask
#     return MaskedData[ScatteringRunType](data)


# providers = (apply_pixel_masks, to_pixel_mask)


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
    # LogicalDimsData,
    MaskedData,
    MaskedDetectorIDs,
    PixelMask,
    PixelMaskFilename,
    SampleRun,
    ScatteringRunType,
    TofData,
)


# def to_pixel_mask(data: TofData[SampleRun], masked: MaskedDetectorIDs) -> PixelMask:
#     """Convert a list of masked detector IDs to a pixel mask.

#     Parameters
#     ----------
#     data:
#         Raw data, defining the detector IDs.
#     masked:
#         The masked detector IDs.
#     """
#     ids = data.coords['detector_number']
#     mask = sc.zeros(sizes=ids.sizes, dtype='bool')
#     mask.values[np.isin(ids.values, masked.values)] = True
#     return PixelMask(mask)


def apply_pixel_masks(
    data: TofData[ScatteringRunType],
    # masks: Optional[sciline.Series[PixelMaskFilename, PixelMask]],
    masked_ids: Optional[sciline.Series[PixelMaskFilename, MaskedDetectorIDs]],
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
    if masked_ids is not None:
        data = data.copy(deep=False)
        ids = data.coords[
            'detector_number' if 'detector_number' in data.coords else 'detector_id'
        ]
        for name, masked in masked_ids.items():
            mask = sc.zeros(sizes=ids.sizes, dtype='bool')
            mask.values[np.isin(ids.values, masked.values)] = True
            data.masks[name] = mask
    return MaskedData[ScatteringRunType](data)


providers = (apply_pixel_masks,)
