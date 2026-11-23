# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Generic masking functions for POWTEX."""

from typing import cast

import scipp as sc

from .types import (
    CorrectedDetector,
    RunType,
    TofMask,
    TwoThetaMask,
    WavelengthDetector,
    WavelengthMask,
)


def apply_masks(
    data: WavelengthDetector[RunType],
    tof_mask_func: TofMask,
    wavelength_mask_func: WavelengthMask,
    two_theta_mask_func: TwoThetaMask,
) -> sc.DataArray:
    """Apply event masks to detector data."""
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


def add_masks(
    da: WavelengthDetector[RunType],
    tof_mask_func: TofMask,
    wavelength_mask_func: WavelengthMask,
    two_theta_mask_func: TwoThetaMask,
) -> CorrectedDetector[RunType]:
    """Add POWTEX event masks without Lorentz correction to detector data."""
    if da is None:
        return cast(CorrectedDetector[RunType], None)
    masked = apply_masks(
        data=da,
        tof_mask_func=tof_mask_func,
        wavelength_mask_func=wavelength_mask_func,
        two_theta_mask_func=two_theta_mask_func,
    )
    return CorrectedDetector[RunType](masked)


providers = (add_masks,)

__all__ = [
    "add_masks",
    "apply_masks",
]
