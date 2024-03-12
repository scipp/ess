# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""
Components for DREAM
"""
import importlib.metadata

from . import data, detector
from .instrument_view import instrument_view
from .io import fold_nexus_detectors, load_geant4_csv, load_nexus

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

providers = (*detector.providers,)

__all__ = [
    'data',
    'detector',
    'fold_nexus_detectors',
    'instrument_view',
    'load_geant4_csv',
    'load_nexus',
]
