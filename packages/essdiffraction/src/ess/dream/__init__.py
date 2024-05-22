# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""
Components for DREAM
"""

import importlib.metadata

from . import data
from .instrument_view import instrument_view
from .io import load_geant4_csv, nexus

try:
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

providers = (*nexus.providers,)

__all__ = [
    'data',
    'beamline',
    'instrument_view',
    'load_geant4_csv',
    'nexus',
]
