# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""Default parameters and providers for DREAM workflows."""

from __future__ import annotations

from ess.powder.types import (
    BackgroundRun,
    CalibrationFilename,
    DspacingBins,
    Filename,
    IofDspacingTwoTheta,
    IofTof,
    MonitorFilename,
    NeXusDetectorName,
    PixelMaskFilename,
    ReducedTofCIF,
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
    parameter_registry,
)

from .beamline import InstrumentConfiguration

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
parameter_registry[InstrumentConfiguration] = ParamWithOptions.from_enum(
    InstrumentConfiguration, default=InstrumentConfiguration.high_flux
)
parameter_registry[MonitorFilename[SampleRun]] = FilenameParameter.from_type(
    MonitorFilename[SampleRun]
)
parameter_registry[MonitorFilename[VanadiumRun]] = FilenameParameter.from_type(
    MonitorFilename[VanadiumRun]
)
parameter_registry[PixelMaskFilename] = MultiFilenameParameter.from_type(
    PixelMaskFilename,
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

# TODO: the mask parameters (TofMask, TwoThetaMask, WavelengthMask) need a new widget
# that allows to define a python function in the notebook and pass it to the workflow.
# We defer this to later; the masks are set to None by default in the workflow for now.

typical_outputs = (IofTof, IofDspacingTwoTheta, ReducedTofCIF)
