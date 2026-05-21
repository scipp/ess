# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from ..reflectometry.corrections import correct_by_footprint, correct_by_proton_charge
from ..reflectometry.types import (
    BeamDivergenceLimits,
    CoordTransformationGraph,
    ProtonCharge,
    RawDetector,
    ReducibleData,
    RunType,
    WavelengthBins,
    YIndexLimits,
    ZIndexLimits,
)
from .conversions import add_coords, add_masks


def add_coords_masks_and_apply_corrections(
    da: RawDetector[RunType],
    ylim: YIndexLimits,
    zlims: ZIndexLimits,
    bdlim: BeamDivergenceLimits,
    wbins: WavelengthBins,
    proton_charge: ProtonCharge[RunType],
    graph: CoordTransformationGraph[RunType],
) -> ReducibleData[RunType]:
    """
    Computes coordinates, masks and corrections that are
    the same for the sample measurement and the reference measurement.
    """
    da = add_coords(da, graph)
    da = add_masks(da, ylim, zlims, bdlim, wbins)

    da = correct_by_footprint(da)

    # For some older Amor files there are no entries in the proton charge log
    if len(proton_charge) != 0:
        da = correct_by_proton_charge(da, proton_charge=proton_charge)

    return ReducibleData[RunType](da)


providers = (add_coords_masks_and_apply_corrections,)
