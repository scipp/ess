# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
Masking functions for the powder workflow.
"""

from collections.abc import Iterable

import numpy as np
import sciline
import scipp as sc

from .types import (
    MaskedData,
    MaskedDetectorIDs,
    NormalizedRunData,
    PixelMaskFilename,
    RunType,
    TofMask,
    TwoThetaMask,
    WavelengthMask,
)


def read_pixel_masks(filename: PixelMaskFilename) -> MaskedDetectorIDs:
    """Read a pixel mask from a Scipp hdf5 file.

    Parameters
    ----------
    filename:
        Path to the hdf5 file.
    """
    return MaskedDetectorIDs({filename: sc.io.load_hdf5(filename)})


def apply_masks(
    data: NormalizedRunData[RunType],
    masked_pixel_ids: MaskedDetectorIDs,
    tof_mask_func: TofMask,
    wavelength_mask_func: WavelengthMask,
    two_theta_mask_func: TwoThetaMask,
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


def _merge(*dicts: dict) -> dict:
    return {key: value for d in dicts for key, value in d.items()}


def with_pixel_mask_filenames(
    workflow: sciline.Pipeline, masks: Iterable[str]
) -> sciline.Pipeline:
    """
    Return modified workflow with pixel mask filenames set.

    Parameters
    ----------
    workflow:
        Workflow to modify.
    masks:
        List or tuple of pixel mask filenames to set.
    """
    workflow = workflow.copy()
    # Workaround bug in Cyclebane, which does not allow empty maps
    if len(masks) == 0:
        workflow[MaskedDetectorIDs] = MaskedDetectorIDs({})
        return workflow
    workflow[MaskedDetectorIDs] = (
        workflow[MaskedDetectorIDs].map({PixelMaskFilename: masks}).reduce(func=_merge)
    )
    return workflow
