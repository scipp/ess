# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NewType

import sciline as sl
import scipp as sc
from scippneutron.tof import chopper_cascade

from ..nexus.types import Component, MonitorType, RunType

LookupTableFilename = NewType("LookupTableFilename", str)
"""Filename of the wavelength lookup table."""


LtotalRange = NewType("LtotalRange", tuple[sc.Variable, sc.Variable])
"""
Range (min, max) of the total length of the flight path from the source to the detector.
This is used to create the lookup table to compute the neutron time-of-flight.
Note that the resulting table will extend slightly beyond this range, as the supplied
range is not necessarily a multiple of the distance resolution.

Note also that the range of total flight paths is supplied manually to the workflow
instead of being read from the input data, as it allows us to compute the expensive part
of the workflow in advance (the lookup table) and does not need to be repeated for each
run, or for new data coming in in the case of live data collection.
"""

DistanceResolution = NewType("DistanceResolution", sc.Variable)
"""
Step size of the distance axis in the lookup table.
Should be a single scalar value with a unit of length.
This is typically of the order of 1-10 cm.
"""


@dataclass
class SourcePulse:
    time: tuple[sc.Variable, sc.Variable]
    """Time range (start, end) of the source pulse."""
    wavelength: tuple[sc.Variable, sc.Variable]
    """Wavelength range (min, max) of the neutrons in the source pulse."""


TimeResolution = NewType("TimeResolution", sc.Variable)
"""
Step size of the event_time_offset axis in the lookup table.
This is basically the 'time-of-flight' resolution of the detector.
Should be a single scalar value with a unit of time.
This is typically of the order of 0.1-0.5 ms.

Since the event_time_offset range needs to span exactly one pulse period, the final
resolution in the lookup table will be at least the supplied value here, but may be
smaller if the pulse period is not an integer multiple of the time resolution.
"""

PulsePeriod = NewType("PulsePeriod", sc.Variable)
"""
Period of the source pulses, i.e., time between consecutive pulse starts.
"""

PulseStride = NewType("PulseStride", int)
"""
Stride of used pulses. Usually 1, but may be a small integer when pulse-skipping.
"""

SourcePosition = NewType("SourcePosition", sc.Variable)
"""
Position of the neutron source in the coordinate system of the choppers.
"""

ChopperFrameSequence = NewType("ChopperFrameSequence", chopper_cascade.FrameSequence)
"""
Sequence of chopper frames used to compute the wavelength as a function of distance and
event_time_offset in the lookup table.
"""


@dataclass
class LookupTable:
    """
    Lookup table giving wavelength as a function of distance and ``event_time_offset``.
    """

    array: sc.DataArray
    """The lookup table data array that maps (distance, event_time_offset) to
    wavelength."""
    pulse_period: sc.Variable
    """Pulse period of the neutron source."""
    pulse_stride: int
    """Pulse stride used when generating the lookup table."""
    distance_resolution: sc.Variable
    """Resolution of the distance coordinate in the lookup table."""
    time_resolution: sc.Variable
    """Resolution of the event_time_offset coordinate in the lookup table."""
    choppers: sc.DataGroup | None = None
    """Chopper parameters used when generating the lookup table, if any. This is made
    optional so we can still support old lookup tables without chopper info."""

    def save_hdf5(self, filename: str | Path) -> None:
        """Save the lookup table to an HDF5 file."""
        sc.DataGroup(asdict(self)).save_hdf5(filename)

    def plot(self, *args, **kwargs) -> Any:
        """Plot the data array of the lookup table."""
        return self.array.plot(*args, **kwargs)


class ErrorLimitedLookupTable(sl.Scope[Component, LookupTable], LookupTable):
    """Lookup table that is masked with NaNs in regions where the standard deviation of
    the wavelength is above a certain threshold."""


PulseStrideOffset = NewType("PulseStrideOffset", int | None)
"""
When pulse-skipping, the offset of the first pulse in the stride. This is typically
zero but can be a small integer < pulse_stride. If None, a guess is made.
"""

LookupTableRelativeErrorThreshold = NewType("LookupTableRelativeErrorThreshold", dict)
"""
Threshold for the relative standard deviation (coefficient of variation) of the
projected wavelength above which values are masked.
The threshold can be different for different beamline components (monitors, detector
banks, etc.). The dictionary should have the component names as keys and the
corresponding thresholds as values.

Example:

.. code-block:: python

   workflow[LookupTableRelativeErrorThreshold] = {
       'detector': 0.1,
       'monitor_close_to_source': 1.0,
       'monitor_far_from_source': 0.2,
   }
"""


class DetectorLtotal(sl.Scope[RunType, sc.Variable], sc.Variable):
    """Total path length of neutrons from source to detector (L1 + L2)."""


class MonitorLtotal(sl.Scope[RunType, MonitorType, sc.Variable], sc.Variable):
    """Total path length of neutrons from source to monitor."""


class WavelengthDetector(sl.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data with wavelength coordinate."""


class WavelengthMonitor(sl.Scope[RunType, MonitorType, sc.DataArray], sc.DataArray):
    """Monitor data with wavelength coordinate."""
