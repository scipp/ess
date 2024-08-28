# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import NewType

import scipp as sc

ImageDetectorName = NewType('ImageDetectorName', str)
"""Histogram mode detector name."""

ImageKeyLogs = NewType('ImageKeyLogs', sc.DataArray)
"""Image key logs."""

RotationMotionSensorName = NewType('RotationMotionSensorName', str)
"""Rotation sensor name."""

RotationLogs = NewType('RotationLogs', sc.DataArray)
"""Rotation logs data."""

HistogramModeDetectorsPath = NewType('HistogramModeDetectorsPath', str)
"""Path to the histogram mode detectors in a nexus file."""

DEFAULT_HISTOGRAM_PATH = HistogramModeDetectorsPath(
    "/entry/instrument/histogram_mode_detectors"
)
