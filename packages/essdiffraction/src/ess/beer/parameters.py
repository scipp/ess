# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""Default parameter specs for BEER workflows."""

from __future__ import annotations

from math import pi

from ess.powder.masking import with_pixel_mask_filenames
from ess.powder.types import (
    CalibrationFilename,
    DspacingBins,
    EmptyCanRun,
    Filename,
    IntensityDspacing,
    IntensityDspacingTwoTheta,
    MaskedDetectorIDs,
    MonitorFilename,
    NeXusDetectorName,
    PixelMaskFilename,
    ReducedTofCIF,
    SampleRun,
    TwoThetaBins,
    UncertaintyBroadcastMode,
    VanadiumRun,
)

from ess.reduce.parameter import ParameterRegistry, ParameterSpec
from ess.reduce.parameter_models import AngleUnit, DspacingEdges, TwoTheta

from .types import DetectorBank


def _edges(model):
    return model.get_edges()


parameters = ParameterRegistry()

parameters[Filename[SampleRun]] = ParameterSpec(
    model=str | None,
    category='Files',
    title='Sample Run',
    default=None,
)
parameters[Filename[VanadiumRun]] = ParameterSpec(
    model=str | None,
    category='Files',
    title='Vanadium Run',
    default=None,
)
parameters[Filename[EmptyCanRun]] = ParameterSpec(
    model=str | None,
    category='Files',
    title='Empty Can Run',
    default=None,
)
parameters[CalibrationFilename] = ParameterSpec(
    model=str | None,
    category='Files',
    title='Calibration',
    default=None,
)
parameters[MonitorFilename[SampleRun]] = ParameterSpec(
    model=str | None,
    category='Files',
    title='Sample Monitor',
    default=None,
)
parameters[PixelMaskFilename] = ParameterSpec(
    model=tuple[str, ...],
    category='Files',
    title='Pixel Masks',
    default=(),
    apply=with_pixel_mask_filenames,
    filter_keys=(MaskedDetectorIDs, PixelMaskFilename),
)

parameters[NeXusDetectorName] = ParameterSpec(
    model=str,
    category='NeXus',
    title='Detector',
    default='detector',
)

parameters[DspacingBins] = ParameterSpec(
    model=DspacingEdges,
    category='Binning',
    title='D-spacing Edges',
    default=DspacingEdges(start=0.0, stop=2.0, num_bins=200),
    transform=_edges,
    use_workflow_default=False,
)
parameters[TwoThetaBins] = ParameterSpec(
    model=TwoTheta,
    category='Binning',
    title='Two-Theta Edges',
    default=TwoTheta(start=0.0, stop=pi, num_bins=180, unit=AngleUnit.RADIAN),
    transform=_edges,
    use_workflow_default=False,
)

parameters[UncertaintyBroadcastMode] = ParameterSpec(
    model=UncertaintyBroadcastMode,
    category='Reduction',
    title='Uncertainty Broadcast',
    default=UncertaintyBroadcastMode.upper_bound,
)
parameters[DetectorBank] = ParameterSpec(
    model=DetectorBank,
    category='Reduction',
    title='Detector Bank',
    default=DetectorBank.south,
)


typical_outputs = (
    IntensityDspacing[SampleRun],
    IntensityDspacingTwoTheta[SampleRun],
    ReducedTofCIF,
)
