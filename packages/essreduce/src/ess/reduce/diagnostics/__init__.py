# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Utilities for inspecting detector and monitor data, when trying to inspect or debug
data recorded during experiments.
"""

import lazy_loader as lazy

from .load_nexus import load_detectors, load_monitors

__getattr__, __dir__, _ = lazy.attach(
    __name__,
    submod_attrs={
        'detector_explorer': ['DetectorExplorer'],
    },
)

__all__ = ['DetectorExplorer', 'load_detectors', 'load_monitors']
