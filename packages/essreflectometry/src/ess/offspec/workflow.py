# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import sciline

from ..reflectometry import providers as reflectometry_providers
from ..reflectometry.types import (
    RawDetector,
    ReducibleData,
    ReferenceRun,
    RunType,
    SampleRun,
    WavelengthBins,
)
from . import conversions, load, maskings, normalization
from .corrections import correct_by_monitor
from .maskings import add_masks
from .types import (
    BackgroundMinWavelength,
    CoordTransformationGraph,
    MonitorData,
    NeXusMonitorName,
    SpectrumLimits,
)


def OffspecWorkflow() -> sciline.Pipeline:
    """
    Workflow with default parameters for the Offspec instrument.
    """
    ps = (
        *providers,
        *reflectometry_providers,
        *load.providers,
        *conversions.providers,
        *maskings.providers,
        *normalization.providers,
    )
    return sciline.Pipeline(
        providers=ps,
        params={NeXusMonitorName: 'monitor2'},
        constraints={RunType: [SampleRun, ReferenceRun]},
    )


def add_coords_masks_and_apply_corrections(
    da: RawDetector[RunType],
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


providers = (add_coords_masks_and_apply_corrections,)
