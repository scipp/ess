# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Domain types for POWTEX powder workflows."""

from collections.abc import Callable
from pathlib import Path
from typing import NewType, TypeVar

import sciline
import scipp as sc

from ess.powder import types as powder_t
from ess.reduce.unwrap import types as unwrap_t

AccumulatedProtonCharge = powder_t.AccumulatedProtonCharge
CalibrationData = powder_t.CalibrationData
CorrectedDetector = powder_t.CorrectedDetector
CorrectedDspacing = powder_t.CorrectedDspacing
DspacingDetector = powder_t.DspacingDetector
EmptyCanRun = powder_t.EmptyCanRun
NormalizedDspacing = powder_t.NormalizedDspacing
Position = powder_t.Position
RawDataAndMetadata = powder_t.RawDataAndMetadata
SampleRun = powder_t.SampleRun
TofMask = powder_t.TofMask
TwoThetaMask = powder_t.TwoThetaMask
UncertaintyBroadcastMode = powder_t.UncertaintyBroadcastMode
VanadiumRun = powder_t.VanadiumRun
WavelengthMask = powder_t.WavelengthMask

TofDetector = unwrap_t.TofDetector
TofMonitor = unwrap_t.TofMonitor
WavelengthDetector = unwrap_t.WavelengthDetector

EmptyVanadiumCanRun = NewType("EmptyVanadiumCanRun", int)
"""Empty-can run for a vanadium measurement."""

CalibrationRun = NewType("CalibrationRun", int)
"""Calibration run."""

RunType = TypeVar("RunType", SampleRun, EmptyCanRun, VanadiumRun, EmptyVanadiumCanRun)
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

KeepEventsMLZ = NewType("KeepEventsMLZ", bool)
"""Flag indicating whether focussed P2D data should keep events instead of histogramming."""


class ShiftNegativeIntensity(sciline.Scope[NonEmptyCanRunType, int], int):
    """Flag controlling whether negative intensity values are shifted."""


class NegativeIntensityResetValue(sciline.Scope[NonEmptyCanRunType, float], float):
    """Replacement value for negative intensities when shifting is disabled."""


class FilenameMLZ(sciline.Scope[GeneralRunType, Path], Path):
    """Filename for local MLZ/POWTEX data."""


class FilepathMLZ(sciline.Scope[GeneralRunType, Path], Path):
    """Directory containing local MLZ/POWTEX data."""


del sc, sciline, NewType, TypeVar
