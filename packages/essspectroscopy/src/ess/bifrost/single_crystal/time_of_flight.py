# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Utilities for computing real neutron time-of-flight."""

import scippnexus as snx

from ess.reduce import unwrap as reduce_unwrap
from ess.reduce.unwrap.types import DetectorLtotal
from ess.spectroscopy.types import (
    DataGroupedByRotation,
    ErrorLimitedLookupTable,
    PulseStrideOffset,
    RawDetector,
    RunType,
    WavelengthDetector,
)


def detector_wavelength_data(
    sample_data: DataGroupedByRotation[RunType],
    lookup: ErrorLimitedLookupTable[snx.NXdetector],
    ltotal: DetectorLtotal[RunType],
    pulse_stride_offset: PulseStrideOffset,
) -> WavelengthDetector[RunType]:
    """
    Convert the arrival-time data to wavelength data using a lookup table.

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
    )


providers = (detector_wavelength_data,)
