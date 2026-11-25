# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Utilities for computing real neutron time-of-flight."""

import scipp as sc
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

# 1 mm is 1% of the lookup table's default `DistanceResolution` (0.1 m) and 6 ppm of
# the 162.8 m flight path, so a variation below it does not change the wavelengths.
# Rounding errors in the transformation chains are orders of magnitude smaller.
_LTOTAL_TOLERANCE = sc.scalar(1e-3, unit='m')


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
    about the vertical axis through the sample, so ``Ltotal`` is nevertheless constant
    along it. This is checked and 'time' is dropped, rather than mapped to the 'a4'
    dimension of the grouped data, which would only add a dimension along which
    nothing varies.

    Raises
    ------
    ValueError
        If ``Ltotal`` changes over time.
    """
    ltotal = detector_ltotal_from_straight_line_approximation(
        detector=monitor,
        source_position=source_position,
        sample_position=sample_position,
        gravity=gravity,
    )
    if 'time' not in ltotal.dims:
        return ltotal
    spread = ltotal.max('time') - ltotal.min('time')
    if (spread > _LTOTAL_TOLERANCE.to(unit=spread.unit)).any().value:
        raise ValueError(
            "The flight path of the Bragg peak monitor changes with the tank angle "
            f"(by up to {spread.max().to(unit='mm').value:.3g} mm), which this "
            "workflow does not support."
        )
    return DetectorLtotal[RunType](ltotal['time', 0])


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
