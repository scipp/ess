# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""
Components for DREAM
"""

import importlib.metadata

from .beamline import InstrumentConfiguration
from .instrument_view import instrument_view
from .io import load_geant4_csv
from .workflows import (
    DreamGeant4MonitorHistogramWorkflow,
    DreamGeant4MonitorIntegratedWorkflow,
    DreamGeant4ProtonChargeWorkflow,
    DreamGeant4Workflow,
    DreamPowderWorkflow,
    DreamWorkflow,
)

try:
    __version__ = importlib.metadata.version("essdiffraction")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

__all__ = [
    'DreamGeant4MonitorHistogramWorkflow',
    'DreamGeant4MonitorIntegratedWorkflow',
    'DreamGeant4ProtonChargeWorkflow',
    'DreamGeant4Workflow',
    'DreamPowderWorkflow',
    'DreamWorkflow',
    'InstrumentConfiguration',
    '__version__',
    'instrument_view',
    'load_geant4_csv',
]
