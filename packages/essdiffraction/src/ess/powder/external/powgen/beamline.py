# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Beamline parameters and utilities for POWGEN.
"""

import scipp as sc

from ...types import (
    CalibrationData,
    FlatDetectorData,
    RawCalibrationData,
    RawDetectorData,
    RunType,
)
from .types import DetectorInfo


def map_detector_to_spectrum(
    data: sc.Dataset, *, detector_info: sc.Dataset
) -> sc.Dataset:
    """
    Transform 'detector' coords to 'spectrum'.

    Parameters
    ----------
    data:
        Input data whose 'detector' coord is transformed.
    detector_info:
        Defines mapping from detector numbers to spectra.

    Returns
    -------
    :
        `data` with 'detector' coord and dim replaced by 'spectrum'.
    """
    if not sc.identical(
        data.coords['detector'].to(
            dtype=detector_info.coords['detector'].dtype, copy=False
        ),
        detector_info.coords['detector'],
    ):
        raise sc.CoordError(
            "The 'detector' coords of `data` and `detector_info` do not match."
        )

    out = data.copy(deep=False)
    del out.coords['detector']
    # Add 1 because spectrum numbers in the data start at 1 but
    # detector_info contains spectrum indices which start at 0.
    out.coords['spectrum'] = detector_info.coords['spectrum'] + sc.index(
        1, dtype=detector_info.coords['spectrum'].dtype
    )

    return out.rename_dims({'detector': 'spectrum'})


def preprocess_calibration_data(
    data: RawCalibrationData, detector_info: DetectorInfo
) -> CalibrationData:
    """Convert calibration data to a format that can be used by Scipp.

    The raw calibration data is encoded in terms of a `'detector'` coordinate.
    This needs to be converted to a `'spectrum'` coordinate to align
    if with sample data.
    """
    return CalibrationData(map_detector_to_spectrum(data, detector_info=detector_info))


def flatten_detector_dimensions(
    data: RawDetectorData[RunType],
) -> FlatDetectorData[RunType]:
    """Flatten logical detector dimensions to a single ``spectrum``.

    POWGEN is loaded with a flat spectrum dimension to begin with,
    so this function does nothing.

    Parameters
    ----------
    data:
        Raw detector data in logical dimensions.

    Returns
    -------
    :
        Flattened detector data with a ``spectrum`` dimension.
    """
    return FlatDetectorData[RunType](data)


providers = (flatten_detector_dimensions, preprocess_calibration_data)
"""Sciline providers for POWGEN beamline processing."""
