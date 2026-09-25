# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import scipp as sc


def add_roi_masks(
    da: sc.DataArray,
    roi: dict,
) -> sc.DataArray:
    """Mask events outside the ROI.
    When a coordinate with the same name exists both on the bins
    and on the events the method masks the event coordinate.
    """
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
    return da.assign_masks(masks).bins.assign_masks(event_masks)
