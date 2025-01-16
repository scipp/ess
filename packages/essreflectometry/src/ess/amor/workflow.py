from ..reflectometry.corrections import correct_by_footprint
from ..reflectometry.types import (
    BeamDivergenceLimits,
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
    graph: CoordTransformationGraph,
) -> ReducibleData[RunType]:
    """
    Computes coordinates, masks and corrections that are
    the same for the sample measurement and the reference measurement.
    """
    da = add_coords(da, graph)
    da = add_masks(da, ylim, zlims, bdlim, wbins)
    correct_by_footprint(da)
    return da


providers = (add_coords_masks_and_apply_corrections,)
