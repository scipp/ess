# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""This module defines the domain types used in ess.powder.

The domain types are used to define parameters and to request results from a Sciline
pipeline.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, NewType, TypeVar

import sciline
import scipp as sc
from scippneutron.io import cif
from scippneutron.metadata import Person, Software

from ess.reduce.nexus import types as reduce_t
from ess.reduce.time_of_flight import types as tof_t
from ess.reduce.uncertainty import UncertaintyBroadcastMode as _UncertaintyBroadcastMode

EmptyDetector = reduce_t.EmptyDetector
EmptyMonitor = reduce_t.EmptyMonitor
RawDetector = reduce_t.RawDetector
DetectorPositionOffset = reduce_t.DetectorPositionOffset
Filename = reduce_t.Filename
GravityVector = reduce_t.GravityVector
RawMonitor = reduce_t.RawMonitor
MonitorPositionOffset = reduce_t.MonitorPositionOffset
NeXusDetectorName = reduce_t.NeXusDetectorName
NeXusMonitorName = reduce_t.NeXusName
NeXusComponent = reduce_t.NeXusComponent
Position = reduce_t.Position

DetectorBankSizes = reduce_t.DetectorBankSizes

TofDetector = tof_t.TofDetector
TofMonitor = tof_t.TofMonitor
PulseStrideOffset = tof_t.PulseStrideOffset
TimeOfFlightLookupTable = tof_t.TimeOfFlightLookupTable
TimeOfFlightLookupTableFilename = tof_t.TimeOfFlightLookupTableFilename

SampleRun = reduce_t.SampleRun
VanadiumRun = reduce_t.VanadiumRun
EmptyCanRun = NewType("EmptyCanRun", int)

CaveMonitor = reduce_t.CaveMonitor
BunkerMonitor = NewType("BunkerMonitor", int)

RunType = TypeVar("RunType", SampleRun, VanadiumRun, EmptyCanRun)
MonitorType = TypeVar("MonitorType", CaveMonitor, BunkerMonitor)


CalibrationFilename = NewType("CalibrationFilename", str | None)
"""Filename of the instrument calibration file."""

DspacingBins = NewType("DspacingBins", sc.Variable)
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


class WavelengthDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data with scattering coordinates computed for all events: wavelength, 2theta,
    d-spacing."""


class DspacingDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data converted to d-spacing."""


DspacingHistogram = NewType("DspacingHistogram", sc.DataArray)
"""Histogrammed intensity vs d-spacing."""


class ElasticCoordTransformGraph(sciline.Scope[RunType, dict], dict):
    """Graph for transforming coordinates in elastic scattering."""


class FocussedDataDspacing(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Intensity vs d-spacing after focussing pixels."""


class FocussedDataDspacingTwoTheta(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Intensity vs (d-spacing, 2theta) after focussing pixels."""


class IntensityDspacing(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data that has been normalized by a vanadium run."""


class IntensityDspacingTwoTheta(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data that has been normalized by a vanadium run, and grouped into 2theta bins."""


class EmptyCanSubtractedIofDspacing(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Intensity vs. d-spacing, subtracted by empty can measurement."""


class EmptyCanSubtractedIofDspacingTwoTheta(
    sciline.Scope[RunType, sc.DataArray], sc.DataArray
):
    """Intensity vs. d-spacing and 2theta, subtracted by empty can measurement."""


IntensityTof = NewType("IntensityTof", sc.DataArray)
"""Normalized data that has been converted to ToF."""

EmptyCanSubtractedIntensityTof = NewType("EmptyCanSubtractedIntensityTof", sc.DataArray)
"""Normalized and empty-can-subtracted data that has been converted to ToF."""


class CorrectedDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data with masks and applied corrections.

    Corrections can include

    - Lorentz correction
    - Absorption correction
    """


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


class CorrectedDspacing(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data in d-spacing - 2theta space including corrections."""


class NormalizedDspacing(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data in d-spacing - 2theta space normalized by monitor/proton charge."""


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

ReducerSoftware = NewType('ReducerSoftware', list[Software])
"""Pieces of software used to reduce the data."""

Source = reduce_t.Source
"""Neutron source metadata."""

CIFAuthors = NewType('CIFAuthors', list[Person])
"""List of authors to save to output CIF files."""

ReducedTofCIF = NewType("ReducedTofCIF", cif.CIF)
"""Reduced data in time-of-flight, ready to be saved to a CIF file."""

ReducedEmptyCanSubtractedTofCIF = NewType("ReducedEmptyCanSubtractedTofCIF", cif.CIF)
"""Reduced data in time-of-flight, ready to be saved to a CIF file."""


@dataclass(frozen=True)
class KeepEvents(Generic[RunType]):
    """
    Flag indicating whether the workflow should keep all events when focussing data.

    If False, data will be histogrammed when focussing.
    """

    value: bool


del sc, sciline, NewType, TypeVar
