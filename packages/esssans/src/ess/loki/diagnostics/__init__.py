# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

from functools import partial

from ess.reduce.diagnostics import DetectorExplorer

from .instrument_view import InstrumentView, instrument_view
from .load_nexus import load_detectors, load_monitors

LokiBankViewer = partial(DetectorExplorer, show_dims=['tube', 'pixel'])

del partial, DetectorExplorer

__all__ = [
    'InstrumentView',
    'LokiBankViewer',
    'instrument_view',
    'load_detectors',
    'load_monitors',
]
