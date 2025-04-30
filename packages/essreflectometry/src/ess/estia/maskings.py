# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry.types import (
    BeamDivergenceLimits,
    WavelengthBins,
    YIndexLimits,
    ZIndexLimits,
)


def _not_between(v, a, b):
    return (v < a) | (v > b)


def add_masks(
    da: sc.DataArray,
    ylim: YIndexLimits,
    zlims: ZIndexLimits,
    bdlim: BeamDivergenceLimits,
    wbins: WavelengthBins,
) -> sc.DataArray:
    """
    Masks the data by ranges in the detector
    coordinates ``z`` and ``y``, and by the divergence of the beam,
    and by wavelength.
    """
    da = da.assign_masks(
        stripe_range=_not_between(da.coords["stripe"], *ylim),
        z_range=_not_between(da.coords["z_index"], *zlims),
        divergence_too_large=_not_between(
            da.coords["divergence_angle"],
            bdlim[0].to(unit=da.coords["divergence_angle"].unit),
            bdlim[1].to(unit=da.coords["divergence_angle"].unit),
        ),
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
