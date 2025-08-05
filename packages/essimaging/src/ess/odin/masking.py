# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Contains the providers to apply masks to detector data.
"""

import scipp as sc

from ..imaging.types import CountsMasked, CountsWavelength, MaskingRules, RunType


def apply_masks(
    da: CountsWavelength[RunType],
    masks: MaskingRules,
) -> CountsMasked[RunType]:
    out = da.copy(deep=False)
    for coord_name, mask in masks.items():
        if (out.bins is not None) and (coord_name in out.bins.coords):
            out.bins.masks[coord_name] = mask(out.bins.coords[coord_name])
        else:
            coord = (
                sc.midpoints(out.coords[coord_name])
                if out.coords.is_edges(coord_name, coord_name)
                else out.coords[coord_name]
            )
            out.masks[coord_name] = mask(coord)
    return CountsMasked[RunType](out)


providers = (apply_masks,)
