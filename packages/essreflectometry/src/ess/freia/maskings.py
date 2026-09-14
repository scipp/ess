# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry.types import RunType, RunUnnormalizedData, WavelengthBins
from .types import DetectorRegionOfInterest, QDetector


def select_events(
    da: QDetector[RunType],
    roi: DetectorRegionOfInterest[RunType],
    wavelength_bins: WavelengthBins,
) -> RunUnnormalizedData[RunType]:
    """Select a peak using pixel or event coordinates and wavelength."""
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
    return RunUnnormalizedData[RunType](
        da.assign_masks(masks).bins.assign_masks(
            event_masks,
            wavelength=~(
                sc.isfinite(wavelength)
                & (wavelength >= wavelength_bins[0].to(unit=wavelength.unit))
                & (wavelength < wavelength_bins[-1].to(unit=wavelength.unit))
            ),
        )
    )


providers = (select_events,)
