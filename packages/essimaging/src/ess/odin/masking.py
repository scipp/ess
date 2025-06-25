# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
""" """

from ..imaging.types import (
    DetectorWavelengthData,
    MaskedDetectorData,
    MaskingRules,
    RunType,
)


def apply_masks(
    da: DetectorWavelengthData[RunType],
    masks: MaskingRules,
) -> MaskedDetectorData[RunType]:
    out = da.copy(deep=False)
    for coord_name, mask in masks.items():
        if (out.bins is not None) and (coord_name in out.bins.coords):
            out.bins.masks[coord_name] = mask(out.bins.coords[coord_name])
        else:
            out.masks[coord_name] = mask(out.coords[coord_name])
    return MaskedDetectorData[RunType](out)


providers = (apply_masks,)
