# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import importlib.metadata

from . import general, io, sans2d, zoom
from .general import (
    DetectorBankOffset,
    MonitorOffset,
    MonitorSpectrumNumber,
    SampleOffset,
    default_parameters,
)
from .io import CalibrationFilename
from .visualization import plot_flat_detector_xy

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

providers = general.providers

del importlib

__all__ = [
    'CalibrationFilename',
    'DetectorBankOffset',
    'MonitorOffset',
    'MonitorSpectrumNumber',
    'SampleOffset',
    'default_parameters',
    'io',
    'plot_flat_detector_xy',
    'providers',
    'sans2d',
    'zoom',
]
