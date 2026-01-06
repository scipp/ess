# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry.conversions import (
    add_proton_current_coord,
    add_proton_current_mask,
)
from ..reflectometry.corrections import correct_by_proton_current
from ..reflectometry.types import (
    BeamDivergenceLimits,
    CoordTransformationGraph,
    CorrectionsToApply,
    ProtonCurrent,
    ReducibleData,
    RunType,
    TofDetector,
    WavelengthBins,
    YIndexLimits,
    ZIndexLimits,
)
from .conversions import add_coords
from .maskings import add_masks


def add_coords_masks_and_apply_corrections(
    da: TofDetector[RunType],
    ylim: YIndexLimits,
    zlims: ZIndexLimits,
    bdlim: BeamDivergenceLimits,
    wbins: WavelengthBins,
    proton_current: ProtonCurrent[RunType],
    graph: CoordTransformationGraph[RunType],
    corrections_to_apply: CorrectionsToApply,
) -> ReducibleData[RunType]:
    """
    Computes coordinates, masks and corrections that are
    the same for the sample measurement and the reference measurement.
    """
    da = add_coords(da, graph)
    da = add_masks(da, ylim, zlims, bdlim, wbins)

    if len(proton_current) != 0:
        da = add_proton_current_coord(da, proton_current)
        da = add_proton_current_mask(da)

    for correction in corrections_to_apply:
        da = correction(da)

    return ReducibleData[RunType](da)


def correct_by_footprint(da: sc.DataArray) -> sc.DataArray:
    """Corrects the data by the size of the footprint on the sample."""
    return da / sc.sin(da.coords['theta'])


default_corrections = {correct_by_proton_current, correct_by_footprint}

providers = (add_coords_masks_and_apply_corrections,)
