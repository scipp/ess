# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import scipp as sc


def add_masks(
    da: sc.DataArray,
    roi: dict,
    wavelength_bins: sc.Variable,
) -> sc.DataArray:
    """Mask events outside the ROI and wavelength range."""
    masks = {}
    event_masks = {}
    for name, (low, high) in roi.items():
        is_event_coord = name in da.bins.coords
        coord = da.bins.coords[name] if is_event_coord else da.coords[name]
        low, high = low.to(unit=coord.unit), high.to(unit=coord.unit)
        if not (low <= high).value:
            raise ValueError(f'Reversed ROI bounds for {name!r}.')
        (event_masks if is_event_coord else masks)[f'roi_{name}'] = ~(
            (coord >= low) & (coord <= high)
        )
    wavelength = da.bins.coords['wavelength']
    return da.assign_masks(masks).bins.assign_masks(
        event_masks,
        wavelength=~(
            sc.isfinite(wavelength)
            & (wavelength >= wavelength_bins[0].to(unit=wavelength.unit))
            & (wavelength < wavelength_bins[-1].to(unit=wavelength.unit))
        ),
    )
