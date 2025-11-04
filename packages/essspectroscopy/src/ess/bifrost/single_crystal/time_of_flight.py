# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Utilities for computing real neutron time-of-flight."""

from ess.reduce import time_of_flight as reduce_time_of_flight
from ess.reduce.time_of_flight.types import DetectorLtotal
from ess.spectroscopy.types import (
    DataGroupedByRotation,
    PulseStrideOffset,
    RawDetector,
    RunType,
    TimeOfFlightLookupTable,
    TofDetector,
)


def detector_time_of_flight_data(
    sample_data: DataGroupedByRotation[RunType],
    lookup: TimeOfFlightLookupTable,
    ltotal: DetectorLtotal[RunType],
    pulse_stride_offset: PulseStrideOffset,
) -> TofDetector[RunType]:
    """
    Convert the time-of-arrival data to time-of-flight data using a lookup table.

    The output data will have a time-of-flight coordinate.

    This is a wrapper around
    :func:`ess.reduce.time_of_flight.detector_time_of_flight_data`
    for different input types.
    """
    return reduce_time_of_flight.eto_to_tof.detector_time_of_flight_data(
        detector_data=RawDetector[RunType](sample_data),
        lookup=lookup,
        ltotal=ltotal,
        pulse_stride_offset=pulse_stride_offset,
    )


providers = (detector_time_of_flight_data,)
