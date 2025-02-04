from ..reflectometry.conversions import (
    add_proton_current_coord,
    add_proton_current_mask,
)
from ..reflectometry.corrections import correct_by_footprint, correct_by_proton_current
from ..reflectometry.types import (
    BeamDivergenceLimits,
    ProtonCurrent,
    RawDetectorData,
    ReducibleData,
    RunType,
    WavelengthBins,
    YIndexLimits,
    ZIndexLimits,
)
from .conversions import add_coords, add_masks
from .types import CoordTransformationGraph


def add_coords_masks_and_apply_corrections(
    da: RawDetectorData[RunType],
    ylim: YIndexLimits,
    zlims: ZIndexLimits,
    bdlim: BeamDivergenceLimits,
    wbins: WavelengthBins,
    proton_current: ProtonCurrent[RunType],
    graph: CoordTransformationGraph,
) -> ReducibleData[RunType]:
    """
    Computes coordinates, masks and corrections that are
    the same for the sample measurement and the reference measurement.
    """
    da = add_coords(da, graph)
    da = add_masks(da, ylim, zlims, bdlim, wbins)

    da = correct_by_footprint(da)

    # For some older Amor files there are no entries in the proton current log
    if len(proton_current) != 0:
        da = add_proton_current_coord(da, proton_current)
        da = add_proton_current_mask(da)
        da = correct_by_proton_current(da)

    return ReducibleData[RunType](da)


providers = (add_coords_masks_and_apply_corrections,)
