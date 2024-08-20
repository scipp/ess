# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import importlib.metadata

from . import general, io, sans2d, zoom
from .general import default_parameters, SampleOffset, MonitorOffset
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
    'MonitorOffset',
    'SampleOffset',
    'io',
    'providers',
    'plot_flat_detector_xy',
    'sans2d',
    'default_parameters',
    'zoom',
]
