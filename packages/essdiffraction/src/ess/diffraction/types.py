# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""This module defines the domain types uses in ess.diffraction.

The domain types are used to define parameters and to request results from a Sciline
pipeline.
"""

from typing import NewType, TypeVar

import sciline
import scipp as sc

# 1 TypeVars used to parametrize the generic parts of the workflow

# 1.1 Run types
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

DspacingBins = NewType('DSpacingBins', sc.Variable)
"""Bin edges for d-spacing."""


class Filename(sciline.Scope[RunType, str], str):
    """Filename of a run."""


TwoThetaBins = NewType('TwoThetaBins', sc.Variable)
"""Bin edges for 2theta."""

ValidTofRange = NewType('ValidTofRange', sc.Variable)
"""Min and max tof value of the instrument."""

# 3 Workflow (intermediate) results


class AccumulatedProtonCharge(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Total proton charge."""


CalibrationData = NewType('CalibrationData', sc.Dataset)
"""Detector calibration data."""

# This is Mantid-specific and can probably be removed when the POWGEN
# workflow is removed.
DetectorInfo = NewType('DetectorInfo', sc.Dataset)
"""Mapping between detector numbers and spectra."""


class DspacingData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data converted to d-spacing."""


class DspacingDataWithoutVariances(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data converted to d-spacing where variances where removed."""


DspacingHistogram = NewType('DspacingHistogram', sc.DataArray)
"""Histogrammed intensity vs d-spacing."""


class FilteredData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data without invalid events."""


class MergedPixels(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Intensity vs d-spacing for all detector pixels combined."""


class NormalizedByProtonCharge(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data that has been normalized by proton charge."""


NormalizedByVanadium = NewType('NormalizedByVanadium', sc.DataArray)
"""Data that has been normalized by a vanadium run."""


class ProtonCharge(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Time-dependent proton charge."""


RawCalibrationData = NewType('RawCalibrationData', sc.Dataset)
"""Calibration data as loaded from file, needs preprocessing before using."""


class RawData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data."""


class RawDataWithvariances(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data that has variances which need special handling."""


class RawDataAndMetadata(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Raw data and associated metadata."""


class TofCroppedData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Raw data cropped to the valid TOF range."""
