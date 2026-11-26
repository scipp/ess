# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Utilities for computing real neutron time-of-flight."""

import scippnexus as snx
from ess.spectroscopy.types import (
    DataGroupedByRotation,
    EmptyDetector,
    ErrorLimitedLookupTable,
    GravityVector,
    NeutronSourcePosition,
    Position,
    PulseStrideOffset,
    RawDetector,
    RunType,
    WavelengthDetector,
)

from ess.reduce import unwrap as reduce_unwrap
from ess.reduce.unwrap.to_wavelength import (
    detector_ltotal_from_straight_line_approximation,
)
from ess.reduce.unwrap.types import DetectorLtotal


def detector_ltotal(
    monitor: EmptyDetector[RunType],
    source_position: NeutronSourcePosition[RunType],
    sample_position: Position[snx.NXsample, RunType],
    gravity: GravityVector,
) -> DetectorLtotal[RunType]:
    """
    Compute Ltotal of the Bragg peak monitor from the straight-line approximation.

    The Bragg peak monitor views the beam direct from the sample, without an analyzer
    in between, so source -> sample -> monitor is the correct flight path.

    Computed from the geometry alone, not from the event data, so that the lookup
    table built from ``Ltotal`` is a static node when the workflow streams.

    The monitor is mounted on the detector tank, so a time-dependent tank angle gives
    the monitor position, and hence ``Ltotal``, a 'time' dimension. The tank rotates
    about the vertical axis through the sample, so ``Ltotal`` is constant along it up
    to survey offsets of the axis, which are far below the lookup table's distance
    resolution. 'time' is therefore averaged out rather than mapped to the 'a4'
    dimension of the grouped data.
    """
    ltotal = detector_ltotal_from_straight_line_approximation(
        detector=monitor,
        source_position=source_position,
        sample_position=sample_position,
        gravity=gravity,
    )
    if 'time' in ltotal.dims:
        return DetectorLtotal[RunType](ltotal.mean('time'))
    return ltotal


def detector_wavelength_data(
    sample_data: DataGroupedByRotation[RunType],
    lookup: ErrorLimitedLookupTable[RunType, snx.NXdetector],
    ltotal: DetectorLtotal[RunType],
    pulse_stride_offset: PulseStrideOffset,
) -> WavelengthDetector[RunType]:
    """
    Convert the time-of-arrival data to wavelength data using a lookup table.

    The output data will have a wavelength coordinate.

    This is a wrapper around
    :func:`ess.reduce.unwrap.detector_wavelength_data`
    for different input types.
    """
    return reduce_unwrap.to_wavelength.detector_wavelength_data(
        detector_data=RawDetector[RunType](sample_data),
        lookup=lookup,
        ltotal=ltotal,
        pulse_stride_offset=pulse_stride_offset,
        keep_event_time_offset=False,
    )


providers = (detector_ltotal, detector_wavelength_data)
