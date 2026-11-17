# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
Masking functions for the powder workflow.
"""

import scipp as sc

from ess.reduce.masking import read_pixel_masks, to_detector_masks

from .types import (
    PixelMasks,
    RunType,
    TofMask,
    TwoThetaMask,
    WavelengthDetector,
    WavelengthMask,
)


def apply_masks(
    data: WavelengthDetector[RunType],
    pixel_masks: PixelMasks,
    tof_mask_func: TofMask,
    wavelength_mask_func: WavelengthMask,
    two_theta_mask_func: TwoThetaMask,
) -> sc.DataArray:
    """ """
    out = data.copy(deep=False)
    if len(pixel_masks) > 0:
        key = (
            set(out.coords.keys()) & {"detector_number", "detector_id", "spectrum"}
        ).pop()
        out = out.assign_masks(to_detector_masks(pixel_masks, out, id_coord=key))

    for dim, mask in {
        "tof": tof_mask_func,
        "wavelength": wavelength_mask_func,
        "two_theta": two_theta_mask_func,
    }.items():
        if mask is not None:
            if (out.bins is not None) and (dim in out.bins.coords):
                out.bins.masks[dim] = mask(out.bins.coords[dim])
            else:
                coord = (
                    sc.midpoints(out.coords[dim])
                    if out.coords.is_edges(dim, dim)
                    else out.coords[dim]
                )
                out.masks[dim] = mask(coord)

    return out


providers = (read_pixel_masks,)
