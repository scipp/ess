# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
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
from .conversions import add_coords, add_masks, validate_wavelength_range
from .corrections import mask_events_where_proton_current_is_too_low
from .types import ChopperFrequency


def add_coords_and_masks(
    da: RawDetector[RunType],
    ylim: YIndexLimits,
    zlims: ZIndexLimits,
    bdlim: BeamDivergenceLimits,
    wbins: WavelengthBins,
    proton_charge: ProtonCharge[RunType],
    graph: CoordTransformationGraph[RunType],
    chopper_frequency: ChopperFrequency[RunType],
) -> ReducibleData[RunType]:
    """
    Computes the coordinates and masks that are
    the same for the sample measurement and the reference measurement.

    The corrections that depend on the kind of reference measurement, and
    the normalization by the proton charge, are applied when the reference
    and the sample are normalized, see :mod:`ess.amor.normalization`.
    The events of several runs can therefore be combined before normalization.
    """
    da = add_coords(da, graph)
    validate_wavelength_range(
        wbins, da.coords['L1'], da.coords['L2'], chopper_frequency
    )
    da = add_masks(da, ylim, zlims, bdlim, wbins)
    da = mask_events_where_proton_current_is_too_low(da, proton_charge)
    return ReducibleData[RunType](da)


providers = (add_coords_and_masks,)
