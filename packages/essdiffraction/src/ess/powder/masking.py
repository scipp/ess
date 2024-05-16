# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
Masking functions for the powder workflow.
"""
from typing import Union, Optional

import numpy as np
import sciline
import scipp as sc

from ess.reduce.masking import mask_range

from .types import (
    DataWithScatteringCoordinates,
    MaskedData,
    MaskedDetectorIDs,
    NormalizedByProtonCharge,
    PixelMaskedData,
    PixelMaskFilename,
    RunType,
    TofMask,
    TofMaskedData,
    TwoThetaData,
    TwoThetaMask,
    TwoThetaMaskedData,
    WavelengthData,
    WavelengthMask,
    WavelengthMaskedData,
)


def read_pixel_masks(
    filename: PixelMaskFilename,
) -> MaskedDetectorIDs:
    """Read a pixel mask from a Scipp hdf5 file.

    Parameters
    ----------
    filename:
        Path to the hdf5 file.
    """
    return MaskedDetectorIDs(sc.io.load_hdf5(filename))


def apply_masks(
    data: DataWithScatteringCoordinates[RunType],
    masked_pixel_ids: sciline.Series[PixelMaskFilename, MaskedDetectorIDs],
    tof_mask_func: Optional[TofMask] = None,
    wavelength_mask_func: Optional[WavelengthMask] = None,
    two_theta_mask_func: Optional[TwoThetaMask] = None,
) -> MaskedData[RunType]:
    """ """
    out = data.copy(deep=False)
    masked_pixel_ids = {"pix_mask": sc.arange("spectrum", 1, 101)}
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


# # def apply_wavelength_masks(
# #     da: WavelengthData[RunType], mask: WavelengthMask
# # ) -> WavelengthMaskedData[RunType]:
# #     if "wavelength" in da.coords and da.coords["wavelength"].ndim > 1:
# #         da = da.bin(wavelength=1)
# #     return WavelengthMaskedData[RunType](mask_range(da, mask=mask))


# # def apply_twotheta_masks(
# #     da: TwoThetaData[RunType], mask: TwoThetaMask
# # ) -> TwoThetaMaskedData[RunType]:
# #     return TwoThetaMaskedData[RunType](mask_range(da, mask=mask))


# def apply_tof_masking(
#     da: DataWithScatteringCoordinates[RunType], mask_func: TofMask
# ) -> TofMaskedData[RunType]:
#     out = da.copy(deep=False)
#     out.masks["tof"] = mask_func(da.coords["tof"])
#     return TofMaskedData[RunType](out)


# def apply_wavelength_masking(
#     da: TofMaskedData[RunType], mask_func: WavelengthMask
# ) -> WavelengthMaskedData[RunType]:
#     out = da.copy(deep=False)
#     out.masks["wavelength"] = mask_func(da.coords["wavelength"])
#     return WavelengthMaskedData[RunType](out)


# def apply_twotheta_masking(
#     da: WavelengthMaskedData[RunType], mask_func: TwoThetaMask
# ) -> TwoThetaMaskedData[RunType]:
#     out = da.copy(deep=False)
#     out.masks["two_theta"] = mask_func(da.coords["two_theta"])
#     return TwoThetaMaskedData[RunType](out)


providers = (
    read_pixel_masks,
    apply_masks,
    # apply_tof_masking,
    # apply_twotheta_masking,
    # apply_wavelength_masking,
)
