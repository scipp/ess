# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""Default parameters and providers for DREAM workflows."""

from __future__ import annotations

from ess.powder.types import (
    BackgroundRun,
    CalibrationFilename,
    DspacingBins,
    Filename,
    IofDspacing,
    IofDspacingTwoTheta,
    NeXusDetectorName,
    PixelMaskFilename,
    ReducedDspacingCIF,
    SampleRun,
    UncertaintyBroadcastMode,
    VanadiumRun,
)
from ess.reduce.parameter import (
    BinEdgesParameter,
    FilenameParameter,
    MultiFilenameParameter,
    ParamWithOptions,
    StringParameter,
    parameter_mappers,
    parameter_registry,
)

parameter_registry[Filename[SampleRun]] = FilenameParameter.from_type(
    Filename[SampleRun]
)
parameter_registry[Filename[VanadiumRun]] = FilenameParameter.from_type(
    Filename[VanadiumRun]
)
parameter_registry[Filename[BackgroundRun]] = FilenameParameter.from_type(
    Filename[BackgroundRun]
)
parameter_registry[CalibrationFilename] = FilenameParameter.from_type(
    CalibrationFilename
)
parameter_mappers[PixelMaskFilename] = MultiFilenameParameter.from_type(
    PixelMaskFilename, default=['']
)
parameter_registry[NeXusDetectorName] = StringParameter.from_type(
    NeXusDetectorName, default="mantle"
)
parameter_registry[DspacingBins] = BinEdgesParameter(
    DspacingBins, dim='dspacing', unit='angstrom', start=0.0, stop=2.0, nbins=200
)
parameter_registry[UncertaintyBroadcastMode] = ParamWithOptions.from_enum(
    UncertaintyBroadcastMode, default=UncertaintyBroadcastMode.upper_bound
)

typical_outputs = (IofDspacing, IofDspacingTwoTheta, ReducedDspacingCIF)
