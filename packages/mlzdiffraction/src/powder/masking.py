# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Masking functions for POWTEX powder reduction."""

import scipp as sc

from .types import (
    DSpacingMask,
    DperpMask,
    ExtendedIntensity2D,
    RunType,
    TofMask,
    TwoThetaMask,
    WavelengthDetector,
    WavelengthMask,
)


def apply_masks_mlz(
    data: WavelengthDetector[RunType],
    tof_mask_func: TofMask,
    wavelength_mask_func: WavelengthMask,
    two_theta_mask_func: TwoThetaMask,
) -> sc.DataArray:
    """Apply event-coordinate masks to detector data."""
    out = data.copy(deep=False)
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


def apply_masks_p2d(
    data: ExtendedIntensity2D,
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


providers = ()
