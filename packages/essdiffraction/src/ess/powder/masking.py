# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
Masking functions for the powder workflow.
"""
import numpy as np
import sciline
import scipp as sc

from ess.reduce.masking import mask_range

from .types import (
    FilePath,
    MaskedDetectorIDs,
    NormalizedByProtonCharge,
    PixelMaskedData,
    PixelMaskFilename,
    RunType,
    TwoThetaData,
    TwoThetaMask,
    TwoThetaMaskedData,
    WavelengthData,
    WavelengthMask,
    WavelengthMaskedData,
)


def read_pixel_masks(
    filename: FilePath[PixelMaskFilename],
) -> MaskedDetectorIDs:
    """Read a pixel mask from a Scipp hdf5 file.

    Parameters
    ----------
    filename:
        Path to the hdf5 file.
    """
    return MaskedDetectorIDs(sc.io.load_hdf5(filename))


def apply_pixel_masks(
    data: NormalizedByProtonCharge[RunType],
    masked_ids: sciline.Series[PixelMaskFilename, MaskedDetectorIDs],
) -> PixelMaskedData[RunType]:
    """Apply pixel-specific masks to raw data.
    The masks are based on detector IDs stored in XML files.

    Parameters
    ----------
    data:
        Raw data with configured component positions.
    masks:
        A series of masks.
    """
    masked_ids = {"pix_mask": sc.arange("spectrum", 1, 101)}
    if len(masked_ids) > 0:
        data = data.copy(deep=False)
        key = (
            set(data.coords.keys()) & {"detector_number", "detector_id", "spectrum"}
        ).pop()
        ids = data.coords[key]
        for name, masked in masked_ids.items():
            mask = sc.zeros(sizes=ids.sizes, dtype="bool")
            mask.values[np.isin(ids.values, masked.values)] = True
            data.masks[name] = mask
    return PixelMaskedData[RunType](data)


def apply_wavelength_masks(
    da: WavelengthData[RunType], mask: WavelengthMask
) -> WavelengthMaskedData[RunType]:
    if "wavelength" in da.coords and da.coords["wavelength"].ndim > 1:
        da = da.bin(wavelength=1)
    return WavelengthMaskedData[RunType](mask_range(da, mask=mask))


def apply_twotheta_masks(
    da: TwoThetaData[RunType], mask: TwoThetaMask
) -> TwoThetaMaskedData[RunType]:
    return TwoThetaMaskedData[RunType](mask_range(da, mask=mask))


providers = (
    read_pixel_masks,
    apply_pixel_masks,
    apply_wavelength_masks,
    apply_twotheta_masks,
)
