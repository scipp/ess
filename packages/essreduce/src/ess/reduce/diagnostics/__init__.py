# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""
Utilities for inspecting detector and monitor data, mainly for when trying to debug or
analyze experimental setups.
"""

from .load import load_detectors, load_monitors

__all__ = ['load_detectors', 'load_monitors']
