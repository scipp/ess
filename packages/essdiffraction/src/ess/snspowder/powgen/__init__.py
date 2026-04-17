# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Functions and classes for the POWGEN instrument.

Note that this module is temporary and will be removed in favor of
the ``dream`` module when that is available.
"""

from . import beamline, peaks
from .calibration import load_calibration
from .instrument_view import instrument_view
from .workflow import PowgenWorkflow, default_parameters

providers = (*beamline.providers,)
"""Sciline Providers for POWGEN-specific functionality."""


__all__ = [
    'PowgenWorkflow',
    'beamline',
    'default_parameters',
    'instrument_view',
    'load_calibration',
    'peaks',
    'providers',
]
