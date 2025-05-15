# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""This module defines the domain types used in ess.powder.

The domain types are used to define parameters and to request results from a Sciline
pipeline.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, NewType, TypeVar

import sciline
import scipp as sc
from scippneutron.io import cif
from scippneutron.metadata import Person, Software

from ess.reduce.nexus import types as reduce_t
from ess.reduce.time_of_flight import types as tof_t
from ess.reduce.uncertainty import UncertaintyBroadcastMode as _UncertaintyBroadcastMode

# 1 TypeVars used to parametrize the generic parts of the workflow

BackgroundRun = reduce_t.BackgroundRun
BunkerMonitor = reduce_t.Monitor2
CalibratedBeamline = reduce_t.CalibratedBeamline
CalibratedDetector = reduce_t.CalibratedDetector
CalibratedBeamline = reduce_t.CalibratedBeamline
CalibratedMonitor = reduce_t.CalibratedMonitor
DetectorData = reduce_t.DetectorData
DetectorPositionOffset = reduce_t.DetectorPositionOffset
EmptyBeamRun = reduce_t.EmptyBeamRun
Filename = reduce_t.Filename
CaveMonitor = reduce_t.Monitor1
MonitorData = reduce_t.MonitorData
MonitorPositionOffset = reduce_t.MonitorPositionOffset
NeXusDetectorName = reduce_t.NeXusDetectorName
NeXusMonitorName = reduce_t.NeXusName
NeXusComponent = reduce_t.NeXusComponent
SampleRun = reduce_t.SampleRun
Position = reduce_t.Position
VanadiumRun = reduce_t.VanadiumRun

DetectorBankSizes = reduce_t.DetectorBankSizes

DetectorTofData = tof_t.DetectorTofData
MonitorTofData = tof_t.MonitorTofData
ResampledMonitorTofData = tof_t.ResampledMonitorTofData
PulsePeriod = tof_t.PulsePeriod
PulseStride = tof_t.PulseStride
PulseStrideOffset = tof_t.PulseStrideOffset
DistanceResolution = tof_t.DistanceResolution
TimeResolution = tof_t.TimeResolution
LtotalRange = tof_t.LtotalRange
LookupTableRelativeErrorThreshold = tof_t.LookupTableRelativeErrorThreshold
TimeOfFlightLookupTable = tof_t.TimeOfFlightLookupTable
TimeOfFlightLookupTableFilename = tof_t.TimeOfFlightLookupTableFilename
SimulationResults = tof_t.SimulationResults

RunType = TypeVar("RunType", SampleRun, VanadiumRun)
MonitorType = TypeVar("MonitorType", CaveMonitor, BunkerMonitor)


# 2 Workflow parameters

CalibrationFilename = NewType("CalibrationFilename", str | None)
"""Filename of the instrument calibration file."""


DspacingBins = NewType("DSpacingBins", sc.Variable)
"""Bin edges for d-spacing."""


OutFilename = NewType("OutFilename", str)
"""Filename of the output."""

TwoThetaBins = NewType("TwoThetaBins", sc.Variable)
"""Bin edges for grouping in 2theta.

This is used by an alternative focussing step that groups detector
pixels by scattering angle into bins given by these edges.
"""

UncertaintyBroadcastMode = _UncertaintyBroadcastMode
"""Mode for broadcasting uncertainties.

See https://doi.org/10.3233/JNR-220049 for context.
"""

ValidTofRange = NewType("ValidTofRange", sc.Variable)
"""Min and max tof value of the instrument."""

# 3 Workflow (intermediate) results


class AccumulatedProtonCharge(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Total proton charge."""

    # Override the docstring of super().__init__ because if contains a broken link
    # when used by Sphinx in ESSdiffraction.
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


CalibrationData = NewType("CalibrationData", sc.Dataset | None)
"""Detector calibration data."""


class DataWithScatteringCoordinates(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data with scattering coordinates computed for all events: wavelength, 2theta,
    d-spacing."""


class DspacingData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data converted to d-spacing."""


DspacingHistogram = NewType("DspacingHistogram", sc.DataArray)
"""Histogrammed intensity vs d-spacing."""

ElasticCoordTransformGraph = NewType("ElasticCoordTransformGraph", dict)
"""Graph for transforming coordinates in elastic scattering."""


class FilteredData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data without invalid events."""


class FocussedDataDspacing(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Intensity vs d-spacing after focussing pixels."""


class FocussedDataDspacingTwoTheta(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Intensity vs (d-spacing, 2theta) after focussing pixels."""


IofDspacing = NewType("IofDspacing", sc.DataArray)
"""Data that has been normalized by a vanadium run."""

IofTof = NewType("IofTof", sc.DataArray)
"""Data that has been normalized by a vanadium run and converted to ToF."""

IofDspacingTwoTheta = NewType("IofDspacingTwoTheta", sc.DataArray)
"""Data that has been normalized by a vanadium run, and grouped into 2theta bins."""


class MaskedData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data with masked pixels, tof regions, wavelength regions, 2theta regions, or
    dspacing regions."""


MaskedDetectorIDs = NewType("MaskedDetectorIDs", dict[str, sc.Variable])
"""1-D variable listing all masked detector IDs."""

CaveMonitorPosition = NewType("CaveMonitorPosition", sc.Variable)
"""Position of DREAM's cave monitor."""


class MonitorFilename(sciline.Scope[RunType, Path], Path):
    """Filename for monitor data.

    Usually, monitors should be stored in the same file as detector data.
    But McStas simulations may output monitors and detectors as separate files.
    """


class WavelengthMonitor(
    sciline.Scope[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Monitor histogram in wavelength."""


class NormalizedRunData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data that has been normalized by proton charge."""


PixelMaskFilename = NewType("PixelMaskFilename", str)
"""Filename of a pixel mask."""


class ProtonCharge(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Time-dependent proton charge."""


class RawDataAndMetadata(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Raw data and associated metadata."""


TofMask = NewType("TofMask", Callable | None)
"""TofMask is a callable that returns a mask for a given TofData."""


TwoThetaMask = NewType("TwoThetaMask", Callable | None)
"""TwoThetaMask is a callable that returns a mask for a given TwoThetaData."""


WavelengthMask = NewType("WavelengthMask", Callable | None)
"""WavelengthMask is a callable that returns a mask for a given WavelengthData."""

Beamline = reduce_t.Beamline
"""Beamline metadata."""

ReducerSoftwares = NewType('ReducerSoftware', list[Software])
"""Pieces of software used to reduce the data."""

Source = reduce_t.Source
"""Neutron source metadata."""

CIFAuthors = NewType('CIFAuthors', list[Person])
"""List of authors to save to output CIF files."""

ReducedTofCIF = NewType("ReducedTofCIF", cif.CIF)
"""Reduced data in time-of-flight, ready to be saved to a CIF file."""


del sc, sciline, NewType, TypeVar
