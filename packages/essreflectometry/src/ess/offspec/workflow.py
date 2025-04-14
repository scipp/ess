# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from ..reflectometry.types import (
    RawDetectorData,
    ReducibleData,
    RunType,
    WavelengthBins,
)
from .corrections import correct_by_monitor
from .maskings import add_masks
from .types import (
    BackgroundMinWavelength,
    CoordTransformationGraph,
    MonitorData,
    SpectrumLimits,
)


def add_coords_masks_and_apply_corrections_direct_beam(
    da: RawDetectorData[RunType],
    spectrum_limits: SpectrumLimits,
    wlims: WavelengthBins,
    wbmin: BackgroundMinWavelength,
    monitor: MonitorData[RunType],
    graph: CoordTransformationGraph[RunType],
) -> ReducibleData[RunType]:
    """
    Computes coordinates, masks and corrections that are
    the same for the sample measurement and the reference measurement.
    """
    da = da.transform_coords(('wavelength',), graph=graph)
    da = add_masks(da, spectrum_limits, wlims)
    da = correct_by_monitor(da, monitor, wlims, wbmin)
    return da


providers = (add_coords_masks_and_apply_corrections_direct_beam,)
