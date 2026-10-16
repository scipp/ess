# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Domain types for POWTEX data reduction workflows."""

from collections.abc import Callable
from pathlib import Path
from typing import NewType, TypeVar

import sciline
import scipp as sc

from ess.reduce.nexus import types as reduce_t
from ess.reduce.uncertainty import UncertaintyBroadcastMode  # noqa: F401
from ess.reduce.unwrap import types as unwrap_t

CalibrationData = NewType("CalibrationData", sc.Dataset | None)
"""Detector calibration data."""
EmptyCanRun = NewType("EmptyCanRun", int)
"""Empty-can run."""
EmptyVanadiumCanRun = NewType("EmptyVanadiumCanRun", int)
"""Empty-can run for a vanadium measurement."""
Position = reduce_t.Position
SampleRun = reduce_t.SampleRun
VanadiumRun = NewType("VanadiumRun", int)
"""Vanadium run."""

RunType = TypeVar("RunType", SampleRun, EmptyCanRun, VanadiumRun, EmptyVanadiumCanRun)


class AccumulatedProtonCharge(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Total proton charge."""


class CorrectedDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data with masks and corrections applied."""


class CorrectedDspacing(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data in d-spacing space after corrections."""


class DspacingDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data with wavelength, two-theta, and d-spacing coordinates."""


class NormalizedDspacing(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data in d-spacing space normalized by a reference run."""


class RawDataAndMetadata(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Raw detector data and associated metadata."""


TofMask = NewType("TofMask", Callable | None)
"""Callable returning a mask for time-of-flight regions."""

TwoThetaMask = NewType("TwoThetaMask", Callable | None)
"""Callable returning a mask for two-theta regions."""

WavelengthMask = NewType("WavelengthMask", Callable | None)
"""Callable returning a mask for wavelength regions."""

TofDetector = unwrap_t.TofDetector
TofMonitor = unwrap_t.TofMonitor
WavelengthDetector = unwrap_t.WavelengthDetector

CalibrationRun = NewType("CalibrationRun", int)
"""Calibration run."""

NonEmptyCanRunType = TypeVar("NonEmptyCanRunType", SampleRun, VanadiumRun)
EmptyCanRunType = TypeVar("EmptyCanRunType", EmptyCanRun, EmptyVanadiumCanRun)
GeneralRunType = TypeVar(
    "GeneralRunType",
    SampleRun,
    EmptyCanRun,
    VanadiumRun,
    EmptyVanadiumCanRun,
    CalibrationRun,
)

KeepEvents = NewType("KeepEvents", bool)
"""Flag indicating whether data should keep events instead of histogramming."""


class ShiftNegativeIntensity(sciline.Scope[NonEmptyCanRunType, int], int):
    """Flag controlling whether negative intensity values are shifted."""


class NegativeIntensityResetValue(sciline.Scope[NonEmptyCanRunType, float], float):
    """Replacement value for negative intensities when shifting is disabled."""


class Filename(sciline.Scope[GeneralRunType, Path], Path):
    """Filename for POWTEX data."""


class Filepath(sciline.Scope[GeneralRunType, Path], Path):
    """Directory containing POWTEX data."""


del sc, sciline, NewType, TypeVar
