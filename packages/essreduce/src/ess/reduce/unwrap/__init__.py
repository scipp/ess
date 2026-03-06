# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""
Utilities for computing neutron wavelength from chopper settings and
neutron time-of-arrival at the detectors.
"""

from ..nexus.types import DiskChoppers
from .lut import (
    BeamlineComponentReading,
    DistanceResolution,
    LookupTableWorkflow,
    LtotalRange,
    NumberOfSimulatedNeutrons,
    PulsePeriod,
    PulseStride,
    SimulationResults,
    SimulationSeed,
    SourcePosition,
    TimeResolution,
    simulate_chopper_cascade_using_tof,
)
from .to_wavelength import providers
from .types import (
    DetectorLtotal,
    ErrorLimitedLookupTable,
    LookupTable,
    LookupTableFilename,
    LookupTableRelativeErrorThreshold,
    MonitorLtotal,
    PulseStrideOffset,
    # ToaDetector,
    # TofDetector,
    # TofLookupTable,
    # TofLookupTableFilename,
    # TofMonitor,
    WavelengthDetector,
    WavelengthMonitor,
)
from .workflow import GenericWavelengthWorkflow

__all__ = [
    "BeamlineComponentReading",
    "DetectorLtotal",
    "DiskChoppers",
    "DistanceResolution",
    "ErrorLimitedLookupTable",
    "GenericWavelengthWorkflow",
    "LookupTable",
    "LookupTableFilename",
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
    "TimeResolution",
    "WavelengthDetector",
    "WavelengthMonitor",
    "providers",
    "simulate_chopper_cascade_using_tof",
]
