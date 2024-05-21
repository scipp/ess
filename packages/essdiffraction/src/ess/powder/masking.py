# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
Masking functions for the powder workflow.
"""
from typing import Optional

import numpy as np
import scipp as sc

from .types import (
    DataWithScatteringCoordinates,
    MaskedData,
    MaskedDetectorIDs,
    PixelMaskFilename,
    RunType,
    TofMask,
    TwoThetaMask,
    WavelengthMask,
)


def read_pixel_masks(
    filename: Optional[PixelMaskFilename] = None,
) -> MaskedDetectorIDs:
    """Read a pixel mask from a Scipp hdf5 file.

    Parameters
    ----------
    filename:
        Path to the hdf5 file.
    """
    masked_ids = {}
    if filename is not None:
        masked_ids = {filename: sc.io.load_hdf5(filename)}
    return MaskedDetectorIDs(masked_ids)


def apply_masks(
    data: DataWithScatteringCoordinates[RunType],
    masked_pixel_ids: MaskedDetectorIDs,
    tof_mask_func: Optional[TofMask] = None,
    wavelength_mask_func: Optional[WavelengthMask] = None,
    two_theta_mask_func: Optional[TwoThetaMask] = None,
) -> MaskedData[RunType]:
    """ """
    out = data.copy(deep=False)
    if len(masked_pixel_ids) > 0:
        key = (
            set(out.coords.keys()) & {"detector_number", "detector_id", "spectrum"}
        ).pop()
        ids = out.coords[key]
        for name, masked in masked_pixel_ids.items():
            mask = sc.zeros(sizes=ids.sizes, dtype="bool")
            mask.values[np.isin(ids.values, masked.values)] = True
            out.masks[name] = mask

    for dim, mask in {
        "tof": tof_mask_func,
        "wavelength": wavelength_mask_func,
        "two_theta": two_theta_mask_func,
    }.items():
        if mask is not None:
            if dim in out.bins.coords:
                out.bins.masks[dim] = mask(out.bins.coords[dim])
            else:
                out.masks[dim] = mask(out.coords[dim])

    return MaskedData[RunType](out)


providers = (read_pixel_masks, apply_masks)
