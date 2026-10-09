# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""
Components for DREAM
"""

import importlib.metadata
from functools import partial

from ess.reduce.nexus import safe_load

from .beamline import InstrumentConfiguration
from .instrument_view import instrument_view
from .io import load_geant4_csv
from .workflows import (
    DETECTOR_BANK_SIZES,
    DreamGeant4MonitorHistogramWorkflow,
    DreamGeant4MonitorIntegratedWorkflow,
    DreamGeant4ProtonChargeWorkflow,
    DreamGeant4Workflow,
    DreamPowderWorkflow,
    DreamWorkflow,
)

load_detectors = partial(safe_load.load_detectors, fold=DETECTOR_BANK_SIZES)
load_monitors = safe_load.load_monitors


try:
    __version__ = importlib.metadata.version("essdiffraction")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib, partial, safe_load, DETECTOR_BANK_SIZES

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
    'load_detectors',
    'load_geant4_csv',
    'load_monitors',
]
