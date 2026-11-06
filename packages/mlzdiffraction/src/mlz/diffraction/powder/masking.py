# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Masking functions for POWTEX powder reduction."""

import scipp as sc

from .types import (
    DperpMask,
    DSpacingMask,
    ExtendedData2D,
    MaskedData2D,
    MaskingData,
    TwoThetaMask,
    WavelengthMask,
)


def apply_masks_p2d(
    data: ExtendedData2D,
    wavelength_mask_func: WavelengthMask,
    two_theta_mask_func: TwoThetaMask,
    dspacing_mask_func: DSpacingMask,
    dperp_mask_func: DperpMask,
) -> sc.Dataset:
    """Apply P2D masks to the extended intensity dataset."""
    out = data.copy(deep=False)
    for name, mask in {
        "lambda": wavelength_mask_func,
        "two_theta": two_theta_mask_func,
        "dspacing_2d": dspacing_mask_func,
        "dperp_2d": dperp_mask_func,
    }.items():
        if mask is not None:
            out[name].masks[f"{name}_mask"] = mask(out[name].data)
    return out


def add_masks_p2d_data(
    data: ExtendedData2D,
    wavelength_mask_func: MaskingData[WavelengthMask],
    two_theta_mask_func: MaskingData[TwoThetaMask],
    dspacing_mask_func: MaskingData[DSpacingMask],
    dperp_mask_func: MaskingData[DperpMask],
) -> MaskedData2D:
    """Apply masks to the extended P2D dataset."""
    return MaskedData2D(
        apply_masks_p2d(
            data,
            wavelength_mask_func,
            two_theta_mask_func,
            dspacing_mask_func,
            dperp_mask_func,
        )
    )


providers = (add_masks_p2d_data,)

__all__ = [
    "add_masks_p2d_data",
    "apply_masks_p2d",
]
