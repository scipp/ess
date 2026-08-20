# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Utilities for computing real neutron time-of-flight."""

import scippnexus as snx
from ess.spectroscopy.types import (
    DataGroupedByRotation,
    EmptyDetector,
    ErrorLimitedLookupTable,
    GravityVector,
    Position,
    PulseStrideOffset,
    RawDetector,
    RunType,
    WavelengthDetector,
)

from ess.reduce import unwrap as reduce_unwrap
from ess.reduce.unwrap.types import DetectorLtotal


def detector_ltotal(
    sample_data: DataGroupedByRotation[RunType],
    source_position: Position[snx.NXsource, RunType],
    sample_position: Position[snx.NXsample, RunType],
    gravity: GravityVector,
) -> DetectorLtotal[RunType]:
    """
    Compute Ltotal from the straight-line approximation.

    The Bragg peak monitor views the beam direct from the sample, without an analyzer
    in between, so a straight line is the correct flight path.

    Computed after grouping so that ``Ltotal`` carries the dimensions of the grouped
    data. BIFROST's tank rotates and the Bragg peak monitor is mounted on it, so the
    monitor position -- and hence ``Ltotal`` -- is time-dependent; ``group_by_rotation``
    turns that 'time' dimension into 'a4'. Deriving ``Ltotal`` from the ungrouped
    geometry instead would leave it labelled with a dimension the data no longer has.

    This is a wrapper around
    :func:`ess.reduce.unwrap.detector_ltotal_from_straight_line_approximation`
    for different input types.
    """
    return reduce_unwrap.to_wavelength.detector_ltotal_from_straight_line_approximation(
        detector=EmptyDetector[RunType](sample_data),
        source_position=source_position,
        sample_position=sample_position,
        gravity=gravity,
    )


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
