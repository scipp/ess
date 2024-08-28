# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 202 Scipp contributors (https://github.com/scipp)
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
    VectorParameter,
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
    IofQ,
    IofQxy,
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
parameter_registry[Filename[SampleRun]] = FilenameParameter.from_type(
    Filename[SampleRun]
)
parameter_registry[Filename[BackgroundRun]] = FilenameParameter.from_type(
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
    WavelengthBins, dim='wavelength', unit='angstrom'
)
parameter_registry[QBins] = BinEdgesParameter(QBins, dim='Q', unit='1/angstrom')
parameter_registry[QxBins] = BinEdgesParameter(QxBins, dim='Qx', unit='1/angstrom')
parameter_registry[QyBins] = BinEdgesParameter(QyBins, dim='Qy', unit='1/angstrom')
parameter_registry[DirectBeam] = StringParameter.from_type(
    DirectBeam, switchable=True, optional=True, default=None
)
parameter_registry[DirectBeamFilename] = FilenameParameter.from_type(
    DirectBeamFilename, switchable=True
)
parameter_registry[BeamCenter] = VectorParameter.from_type(
    BeamCenter, default=sc.vector([0, 0, 0], unit='m')
)

typical_outputs = (
    BackgroundSubtractedIofQ,
    BackgroundSubtractedIofQxy,
    IofQ[SampleRun],
    IofQxy[SampleRun],
    IofQ[BackgroundRun],
    IofQxy[BackgroundRun],
    MaskedData[BackgroundRun],
    MaskedData[SampleRun],
    WavelengthMonitor[SampleRun, Incident],
    WavelengthMonitor[SampleRun, Transmission],
    WavelengthMonitor[BackgroundRun, Incident],
    WavelengthMonitor[BackgroundRun, Transmission],
)
