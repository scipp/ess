# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Utilities for computing real neutron time-of-flight."""

import scippnexus as snx
from ess.spectroscopy.types import (
    DataGroupedByRotation,
    ErrorLimitedLookupTable,
    PulseStrideOffset,
    RawDetector,
    RunType,
    WavelengthDetector,
)

from ess.reduce import unwrap as reduce_unwrap
from ess.reduce.unwrap.types import DetectorLtotal


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
    # A time-dependent detector position (BIFROST's tank rotates, and the Bragg peak
    # monitor is mounted on it) makes ``ltotal`` depend on 'time'. The instrument angle
    # is the only dynamic parameter, so ``group_by_rotation`` has already turned that
    # same 'time' dimension into 'a4'. Rename to match, or the broadcast below rejects
    # ``ltotal`` as having a dimension the data does not.
    if 'time' in ltotal.dims and 'time' not in sample_data.dims:
        ltotal = ltotal.rename_dims(time='a4')
    return reduce_unwrap.to_wavelength.detector_wavelength_data(
        detector_data=RawDetector[RunType](sample_data),
        lookup=lookup,
        ltotal=ltotal,
        pulse_stride_offset=pulse_stride_offset,
    )


providers = (detector_wavelength_data,)
