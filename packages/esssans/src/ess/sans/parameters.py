# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
Default parameters, providers and utility functions for the loki workflow.
"""

from __future__ import annotations

import scipp as sc
from ess.reduce.parameter import (
    BinEdgesParameter,
    BooleanParameter,
    FilenameParameter,
    MultiFilenameParameter,
    ParamWithOptions,
    StringParameter,
    Vector2dParameter,
    parameter_registry,
)

from ..sans.types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    BackgroundSubtractedIofQxy,
    BeamCenter,
    CorrectForGravity,
    DirectBeam,
    DirectBeamFilename,
    EmptyBeamRun,
    Filename,
    Incident,
    IntensityQ,
    IntensityQxQy,
    MaskedData,
    NeXusDetectorName,
    NeXusMonitorName,
    PixelMaskFilename,
    PixelShapePath,
    QBins,
    QxBins,
    QyBins,
    ReturnEvents,
    SampleRun,
    TransformationPath,
    Transmission,
    TransmissionRun,
    UncertaintyBroadcastMode,
    WavelengthBins,
    WavelengthMonitor,
)

parameter_registry[CorrectForGravity] = BooleanParameter.from_type(
    CorrectForGravity, default=False
)
parameter_registry[NeXusDetectorName] = StringParameter.from_type(NeXusDetectorName)

parameter_registry[NeXusMonitorName[Incident]] = StringParameter.from_type(
    NeXusMonitorName[Incident], default=''
)
parameter_registry[NeXusMonitorName[Transmission]] = StringParameter.from_type(
    NeXusMonitorName[Transmission], default=''
)
parameter_registry[TransformationPath] = StringParameter.from_type(
    TransformationPath, default=''
)
parameter_registry[PixelMaskFilename] = MultiFilenameParameter.from_type(
    PixelMaskFilename
)
parameter_registry[PixelShapePath] = StringParameter.from_type(
    PixelShapePath, default=''
)
# Should this be ReductionMode (EventMode/HistogramMode)?
parameter_registry[ReturnEvents] = BooleanParameter.from_type(
    ReturnEvents, default=False
)
parameter_registry[UncertaintyBroadcastMode] = ParamWithOptions.from_enum(
    UncertaintyBroadcastMode, default=UncertaintyBroadcastMode.upper_bound
)
parameter_registry[Filename[SampleRun]] = MultiFilenameParameter.from_type(
    Filename[SampleRun]
)
parameter_registry[Filename[BackgroundRun]] = MultiFilenameParameter.from_type(
    Filename[BackgroundRun]
)
parameter_registry[Filename[TransmissionRun[SampleRun]]] = FilenameParameter.from_type(
    Filename[TransmissionRun[SampleRun]]
)
parameter_registry[Filename[TransmissionRun[BackgroundRun]]] = (
    FilenameParameter.from_type(Filename[TransmissionRun[BackgroundRun]])
)
parameter_registry[Filename[EmptyBeamRun]] = FilenameParameter.from_type(
    Filename[EmptyBeamRun]
)

parameter_registry[WavelengthBins] = BinEdgesParameter(
    WavelengthBins, dim='wavelength', start=2, stop=12.0, nbins=300, log=False
)
parameter_registry[QBins] = BinEdgesParameter(
    QBins, dim='Q', start=0.1, stop=0.3, nbins=100, log=False
)
parameter_registry[QxBins] = BinEdgesParameter(
    QxBins, dim='Qx', start=-0.5, stop=0.5, nbins=100
)
parameter_registry[QyBins] = BinEdgesParameter(
    QyBins, dim='Qy', start=-0.5, stop=0.5, nbins=100
)
parameter_registry[DirectBeam] = StringParameter.from_type(
    DirectBeam, switchable=True, optional=True, default=None
)
parameter_registry[DirectBeamFilename] = FilenameParameter.from_type(
    DirectBeamFilename, switchable=True
)
parameter_registry[BeamCenter] = Vector2dParameter.from_type(
    BeamCenter, default=sc.vector([0, 0, 0], unit='m')
)

typical_outputs = (
    BackgroundSubtractedIofQ,
    BackgroundSubtractedIofQxy,
    IntensityQ[SampleRun],
    IntensityQxQy[SampleRun],
    IntensityQ[BackgroundRun],
    IntensityQxQy[BackgroundRun],
    MaskedData[BackgroundRun],
    MaskedData[SampleRun],
    WavelengthMonitor[SampleRun, Incident],
    WavelengthMonitor[SampleRun, Transmission],
    WavelengthMonitor[BackgroundRun, Incident],
    WavelengthMonitor[BackgroundRun, Transmission],
)
