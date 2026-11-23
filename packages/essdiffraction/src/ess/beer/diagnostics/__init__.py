# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Instrument diagnostics for BEER."""

from .load_nexus import load_detectors, load_monitors

__all__ = [
    'load_detectors',
    'load_monitors',
]
