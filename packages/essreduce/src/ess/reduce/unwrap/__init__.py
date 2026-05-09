# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""
Utilities for computing neutron wavelength from chopper settings and
neutron time-of-arrival at the detectors.
"""

from ..nexus.types import DiskChoppers
from .lut import (
    DistanceResolution,
    LookupTableWorkflow,
    LtotalRange,
    PulsePeriod,
    PulseStride,
    SourcePosition,
    SourcePulse,
    TimeResolution,
)
from .lut_from_simulation import (
    BeamlineComponentReading,
    LookupTableFromSimulation,
    LookupTableFromTof,
    NumberOfSimulatedNeutrons,
    SimulationResults,
    SimulationSeed,
    simulate_chopper_cascade_using_tof,
)
from .to_wavelength import providers
from .types import (
    ChopperFrameSequence,
    DetectorLtotal,
    ErrorLimitedLookupTable,
    LookupTable,
    LookupTableFilename,
    LookupTableRelativeErrorThreshold,
    MonitorLtotal,
    PulseStrideOffset,
    WavelengthDetector,
    WavelengthMonitor,
)
from .workflow import GenericUnwrapWorkflow

__all__ = [
    "BeamlineComponentReading",
    "ChopperFrameSequence",
    "DetectorLtotal",
    "DiskChoppers",
    "DistanceResolution",
    "ErrorLimitedLookupTable",
    "GenericUnwrapWorkflow",
    "LookupTable",
    "LookupTableFilename",
    "LookupTableFromSimulation",
    "LookupTableFromTof",
    "LookupTableRelativeErrorThreshold",
    "LookupTableWorkflow",
    "LtotalRange",
    "MonitorLtotal",
    "NumberOfSimulatedNeutrons",
    "PulsePeriod",
    "PulseStride",
    "PulseStrideOffset",
    "SimulationResults",
    "SimulationSeed",
    "SourcePosition",
    "SourcePulse",
    "TimeResolution",
    "WavelengthDetector",
    "WavelengthMonitor",
    "providers",
    "simulate_chopper_cascade_using_tof",
]
