# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Utilities for inspecting detector and monitor data, when trying to inspect or debug
data recorded during experiments.
"""

from .detector_explorer import DetectorExplorer
from .load_nexus import load_detectors, load_monitors

__all__ = ['DetectorExplorer', 'load_detectors', 'load_monitors']
