# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""This module defines the domain types used in ess.powder.

The domain types are used to define parameters and to request results from a Sciline
pipeline.
"""
from enum import Enum
from pathlib import Path
from typing import NewType, TypeVar

import sciline
import scipp as sc

# 1 TypeVars used to parametrize the generic parts of the workflow

# 1.1 Run types
EmptyCanRun = NewType('EmptyCanRun', int)
"""Empty sample can run."""
EmptyInstrumentRun = NewType('EmptyInstrumentRun', int)
"""Empty instrument run."""
SampleRun = NewType('SampleRun', int)
"""Sample run."""
VanadiumRun = NewType('VanadiumRun', int)
"""Vanadium run."""
RunType = TypeVar('RunType', EmptyInstrumentRun, SampleRun, VanadiumRun)
"""TypeVar used for specifying the run."""


# 2 Workflow parameters

CalibrationFilename = NewType('CalibrationFilename', str)
"""Filename of the instrument calibration file."""


# In Python 3.11, this can be replaced with a StrEnum
class DetectorName(str, Enum):
    """Name of a detector."""

    mantle = 'mantle'
    high_resolution = 'high_resolution'
    endcap_backward = 'endcap_backward'
    endcap_forward = 'endcap_forward'


DspacingBins = NewType('DSpacingBins', sc.Variable)
"""Bin edges for d-spacing."""


class Filename(sciline.Scope[RunType, str], str):
    """Name of an input file."""


class FilePath(sciline.Scope[RunType, Path], Path):
    """Path to an input file on disk."""


OutFilename = NewType('OutFilename', str)
"""Filename of the output."""

TwoThetaBins = NewType('TwoThetaBins', sc.Variable)
"""Bin edges for grouping in 2theta.

This is used by an alternative focussing step that groups detector
pixels by scattering angle into bins given by these edges.
"""

UncertaintyBroadcastMode = Enum(
    'UncertaintyBroadcastMode', ['drop', 'upper_bound', 'fail']
)
"""Mode for broadcasting uncertainties.

See https://doi.org/10.3233/JNR-220049 for context.
"""

ValidTofRange = NewType('ValidTofRange', sc.Variable)
"""Min and max tof value of the instrument."""

# 3 Workflow (intermediate) results


class AccumulatedProtonCharge(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Total proton charge."""


CalibrationData = NewType('CalibrationData', sc.Dataset)
"""Detector calibration data."""


class DetectorDimensions(sciline.Scope[DetectorName, tuple[str, ...]], tuple[str, ...]):
    """Logical detector dimensions."""


class DspacingData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data converted to d-spacing."""


class DspacingDataWithoutVariances(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data converted to d-spacing where variances where removed."""


DspacingHistogram = NewType('DspacingHistogram', sc.DataArray)
"""Histogrammed intensity vs d-spacing."""


class FilteredData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data without invalid events."""


class FocussedData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Intensity vs d-spacing after focussing pixels."""


class NormalizedByProtonCharge(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data that has been normalized by proton charge."""


NormalizedByVanadium = NewType('NormalizedByVanadium', sc.DataArray)
"""Data that has been normalized by a vanadium run."""


class ProtonCharge(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Time-dependent proton charge."""


RawCalibrationData = NewType('RawCalibrationData', sc.Dataset)
"""Calibration data as loaded from file, needs preprocessing before using."""


class RawDataAndMetadata(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Raw data and associated metadata."""


class RawDetector(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Full raw data for a detector."""


class RawDetectorData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data (events / histogram) extracted from a RawDetector."""


class RawSample(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Raw data from a loaded sample."""


RawSource = NewType('RawSource', sc.DataGroup)
"""Raw data from a loaded neutron source."""


class TofCroppedData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data cropped to the valid TOF range."""


del sc, sciline, NewType, TypeVar
