# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry.conversions import (
    add_proton_charge_coord,
    add_proton_charge_mask,
)
from ..reflectometry.corrections import correct_by_proton_charge
from ..reflectometry.types import (
    BeamDivergenceLimits,
    CoordTransformationGraph,
    CorrectionsToApply,
    ProtonCharge,
    ReducibleData,
    RunType,
    WavelengthBins,
    WavelengthDetector,
    YIndexLimits,
    ZIndexLimits,
)
from .conversions import add_coords
from .maskings import add_masks


def add_coords_masks_and_apply_corrections(
    da: WavelengthDetector[RunType],
    ylim: YIndexLimits,
    zlims: ZIndexLimits,
    bdlim: BeamDivergenceLimits,
    wbins: WavelengthBins,
    proton_charge: ProtonCharge[RunType],
    graph: CoordTransformationGraph[RunType],
    corrections_to_apply: CorrectionsToApply,
) -> ReducibleData[RunType]:
    """
    Computes coordinates, masks and corrections that are
    the same for the sample measurement and the reference measurement.
    """
    da = add_coords(da, graph)
    da = add_masks(da, ylim, zlims, bdlim, wbins)

    if len(proton_charge) != 0:
        da = add_proton_charge_coord(da, proton_charge)
        da = add_proton_charge_mask(da)

    for correction in corrections_to_apply:
        da = correction(da)

    return ReducibleData[RunType](da)


def correct_by_footprint(da: sc.DataArray) -> sc.DataArray:
    """Corrects the data by the size of the footprint on the sample."""
    return da / sc.sin(da.coords['theta'])


def assume_time_series_constant_with_zero_default_value_if_empty(da: sc.DataArray):
    '''Converts a time series to a single value by taking the average.
    If the time series if empty it returns the default value 0.'''
    return da.mean() if len(da) > 0 else sc.scalar(0.0, unit=da.unit)


default_corrections = {correct_by_proton_charge, correct_by_footprint}

providers = (add_coords_masks_and_apply_corrections,)
