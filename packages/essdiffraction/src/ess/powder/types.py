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

from ess.packages.essreduce.src.ess.reduce.nexus import types as reduce_t
from ess.reduce.uncertainty import UncertaintyBroadcastMode as _UncertaintyBroadcastMode
from ess.packages.essreduce.src.ess.reduce.unwrap import types as unwrap_t

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
ProtonCharge = reduce_t.ProtonCharge

DetectorBankSizes = reduce_t.DetectorBankSizes

DetectorLtotal = unwrap_t.DetectorLtotal
WavelengthDetector = unwrap_t.WavelengthDetector
WavelengthMonitor = unwrap_t.WavelengthMonitor
TofDetector = unwrap_t.TofDetector
TofMonitor = unwrap_t.TofMonitor
PulseStrideOffset = unwrap_t.PulseStrideOffset
LookupTable = unwrap_t.LookupTable
LookupTableFilename = unwrap_t.LookupTableFilename
LookupTableRelativeErrorThreshold = unwrap_t.LookupTableRelativeErrorThreshold

SampleRun = reduce_t.SampleRun
VanadiumRun = NewType("VanadiumRun", int)
EmptyCanRun = NewType("EmptyCanRun", int)
VanadiumRun = reduce_t.VanadiumRun
EmptyVanadiumCanRun = NewType("EmptyVanadiumCanRun", int)
CalibrationRun = NewType("CalibrationRun", int)

CaveMonitor = NewType("CaveMonitor", int)
BunkerMonitor = NewType("BunkerMonitor", int)

RunType = TypeVar("RunType", SampleRun, EmptyCanRun, VanadiumRun, EmptyVanadiumCanRun)
NonEmptyCanRunType = TypeVar("NonEmptyCanRunType", SampleRun, VanadiumRun)
EmptyCanRunType = TypeVar("EmptyCanRunType", EmptyCanRun, EmptyVanadiumCanRun)
GeneralRunType = TypeVar("GeneralRunType", SampleRun, EmptyCanRun, VanadiumRun, EmptyVanadiumCanRun, CalibrationRun)
MonitorType = TypeVar("MonitorType", CaveMonitor, BunkerMonitor)

TofMask = NewType("TofMask", Callable | None)
"""TofMask is a callable that returns a mask for masking time-of-flight regions."""

TwoThetaMask = NewType("TwoThetaMask", Callable | None)
"""TwoThetaMask is a callable that returns a mask for a given TwoThetaData."""

WavelengthMask = NewType("WavelengthMask", Callable | None)
"""WavelengthMask is a callable that returns a mask for a given WavelengthData."""

DSpacingMask = NewType("DSpacingMask", Callable | None)
"""DSpacingMask is a callable that returns a mask for a given DSpacingData."""

DperpMask = NewType("DperpMask", Callable | None)
"""DperpMask is a callable that returns a mask for a given DperpData."""

MaskedDetectorIDs = NewType("MaskedDetectorIDs", dict[str, sc.Variable])
"""1-D variable listing all masked detector IDs."""

MaskType = TypeVar("MaskType", TwoThetaMask, TofMask, WavelengthMask, DSpacingMask, DperpMask)

CalibrationFilename = NewType("CalibrationFilename", str | None)
"""Filename of the instrument calibration file."""

DspacingBins = NewType("DspacingBins", sc.Variable)
"""Bin edges for d-spacing."""

DspacingNBins = NewType("DspacingNBins", int)
"""Number of d-spacing bins."""

OutFilename = NewType("OutFilename", str)
"""Filename of the output."""

TwoThetaBins = NewType("TwoThetaBins", sc.Variable | None)
"""Bin edges for grouping in 2theta.

This is used by an alternative focussing step that groups detector
pixels by scattering angle into bins given by these edges.

The focussing step aligns its internal, finer 2theta binning with these edges,
so setting them also affects results that are integrated over 2theta.
``None`` means that no 2theta grouping is available: the workflow can then only
produce results integrated over 2theta and raises if grouped data is requested.
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


class DspacingDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data with scattering coordinates computed for all events: wavelength, 2theta,
    d-spacing."""


class QDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """D-spacing detector data with a momentum-transfer coordinate."""


DspacingHistogram = NewType("DspacingHistogram", sc.DataArray)
"""Histogrammed intensity vs d-spacing."""


class DetectorTwoTheta(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Scattering angle (two-theta) for each detector pixel."""


class WavelengthRange(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Minimum and maximum wavelength transmitted by the chopper cascade."""


class ElasticCoordTransformGraph(sciline.Scope[RunType, dict], dict):
    """Graph for transforming coordinates in elastic scattering."""


class ElasticCoordTransformGraphMLZ(sciline.Scope[dict], dict):
    """Graph for transforming coordinates in elastic scattering."""


class MonitorCoordTransformGraph(sciline.Scope[RunType, dict], dict):
    """Graph for transforming coordinates of a monitor."""


class FocussedDataDspacing(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Intensity vs d-spacing after focussing pixels."""


class FocussedDataDspacingTwoTheta(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Intensity vs (d-spacing, 2theta) after focussing pixels."""


class IntensityDspacing(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data that has been normalized by a vanadium run."""


class IntensityDspacingTwoTheta(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data that has been normalized by a vanadium run, and grouped into 2theta bins."""


EmptyCanSubtractedIofDspacing = NewType("EmptyCanSubtractedIofDspacing", sc.DataArray)
"""Intensity vs. d-spacing, subtracted by empty can measurement."""


EmptyCanSubtractedIofDspacingTwoTheta = NewType(
    "EmptyCanSubtractedIofDspacingTwoTheta", sc.DataArray
)
"""Intensity vs. d-spacing and 2theta, subtracted by empty can measurement."""


IntensityTof = NewType("IntensityTof", sc.DataArray)
"""Normalized data that has been converted to ToF."""

NormalizedIntensity2D = NewType("NormalizedIntensity2D", sc.DataArray)
"""2D Intensity of normalized sample data."""

ExtendedIntensity2D = NewType("ExtendedIntensity2D", sc.Dataset)
"""2D Intensity with additional info, including two_theta and lambda."""

MaskedExtendedIntensity2D = NewType("MaskedExtendedIntensity2D", sc.Dataset)
"""2D ExtendedIntensity with masks."""

EmptyCanSubtractedIntensityTof = NewType("EmptyCanSubtractedIntensityTof", sc.DataArray)
"""Normalized and empty-can-subtracted data that has been converted to ToF."""


class SpectralDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """ToDo
    """


class CorrectedDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data with masks and applied corrections.

    Corrections can include

    - Lorentz correction
    - Absorption correction
    """


CaveMonitorPosition = NewType("CaveMonitorPosition", sc.Variable)
"""Position of DREAM's cave monitor."""


class MonitorFilename(sciline.Scope[RunType, Path], Path):
    """Filename for monitor data.

    Usually, monitors should be stored in the same file as detector data.
    But McStas simulations may output monitors and detectors as separate files.
    """


class DspacingMonitor(sciline.Scope[RunType, MonitorType, sc.DataArray], sc.DataArray):
    """Monitor histogram in wavelength."""


class CorrectedDspacing(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data in d-spacing - 2theta space including corrections."""


class Dspacing2D(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data in with d-spacing and dperp dimensions."""


class EmptyCanSubtracted2D(sciline.Scope[NonEmptyCanRunType, sc.DataArray], sc.DataArray):
    """..."""


class EmptyCanSubtractedNoNegative2D(sciline.Scope[NonEmptyCanRunType, sc.DataArray], sc.DataArray):
    """..."""


class ShiftNegativeIntensity(sciline.Scope[NonEmptyCanRunType, int], int):
    """Whether negative intensities should be shifted to positive values by adding the most negative value to all intensities.

    Use 0 for False and 1 for True.
    """


class NegativeIntensityResetValue(sciline.Scope[NonEmptyCanRunType, float], float):
    """The value to which negative intensities should be reset (mapped)."""


class NormalizedDspacing(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data in d-spacing - 2theta space normalized by monitor/proton charge."""


PixelMaskFilename = NewType("PixelMaskFilename", str)
"""Filename of a pixel mask."""


class RawDataAndMetadata(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Raw data and associated metadata."""


class FilenameMLZ(sciline.Scope[GeneralRunType, Path], Path):
    """Filename for MLZ data. Set to 'None' to omit this run type."""


class FilepathMLZ(sciline.Scope[GeneralRunType, Path], Path): ...


class MaskingData(sciline.Scope[MaskType, type], type): ...


Beamline = reduce_t.Beamline
"""Beamline metadata."""

Measurement = reduce_t.Measurement
"""Measurement metadata."""

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

KeepEventsMLZ = NewType("KeepEventsMLZ", bool)
"""Flag indicating whether the workflow should keep all events when focussing data. If False, data will be histogrammed when focussing."""



del sc, sciline, NewType, TypeVar
