# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""This module defines the domain types used in ess.tbl.

The domain types are used to define parameters and to request results from a Sciline
pipeline.
"""

from typing import NewType, TypeVar

import sciline
import scipp as sc

from ess.reduce.nexus import types as reduce_t
from ess.reduce.time_of_flight import types as tof_t

# 1 TypeVars used to parametrize the generic parts of the workflow

BackgroundRun = reduce_t.BackgroundRun
DetectorData = reduce_t.DetectorData
Choppers = reduce_t.Choppers
EmptyBeamRun = reduce_t.EmptyBeamRun
Filename = reduce_t.Filename
Monitor1 = reduce_t.Monitor1
Monitor2 = reduce_t.Monitor2
MonitorData = reduce_t.MonitorData
NeXusDetectorName = reduce_t.NeXusDetectorName
NeXusMonitorName = reduce_t.NeXusName
NeXusComponent = reduce_t.NeXusComponent
SampleRun = reduce_t.SampleRun

DetectorLtotal = tof_t.DetectorLtotal
DetectorTofData = tof_t.DetectorTofData
PulsePeriod = tof_t.PulsePeriod
PulseStride = tof_t.PulseStride
PulseStrideOffset = tof_t.PulseStrideOffset
DistanceResolution = tof_t.DistanceResolution
TimeResolution = tof_t.TimeResolution
LtotalRange = tof_t.LtotalRange
LookupTableRelativeErrorThreshold = tof_t.LookupTableRelativeErrorThreshold
TimeOfFlightLookupTable = tof_t.TimeOfFlightLookupTable
SimulationResults = tof_t.SimulationResults

RunType = TypeVar("RunType", SampleRun, BackgroundRun)
MonitorType = TypeVar("MonitorType", Monitor1, Monitor2)

CoordTransformGraph = NewType("CoordTransformGraph", dict)
"""
Graph of coordinate transformations used to compute the wavelength from the
time-of-flight.
"""


class DetectorWavelengthData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data with wavelength coordinate."""


del sc, sciline, NewType, TypeVar
