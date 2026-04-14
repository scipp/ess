# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry.types import (
    WavelengthBins,
)
from .types import SpectrumLimits


def _not_between(v, a, b):
    return (v < a) | (v > b)


def add_masks(
    da: sc.DataArray,
    spectrum_limits: SpectrumLimits,
    wbins: WavelengthBins,
) -> sc.DataArray:
    """
    Masks the data by range in the detector spectrum and by wavelength.
    """
    da = da.assign_masks(
        not_specularly_reflected_signal=_not_between(
            da.coords['spectrum'], *spectrum_limits
        )
    )
    da = da.bins.assign_masks(
        wavelength=_not_between(
            da.bins.coords['wavelength'],
            wbins[0],
            wbins[-1],
        ),
    )
    return da


providers = ()
