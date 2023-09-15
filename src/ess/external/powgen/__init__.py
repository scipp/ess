# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Functions and classes for the POWGEN instrument.

Note that this module is temporary and will be removed in favor of
the ``dream`` module when that is available.
"""

from . import beamline, data
from .instrument_view import instrument_view

__all__ = [
    'beamline',
    'data',
    'instrument_view',
]
