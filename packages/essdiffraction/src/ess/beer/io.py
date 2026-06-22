# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Backward-compatible import location for BEER McStas helpers."""

from . import mcstas as _mcstas
from .mcstas import *  # noqa: F403
from .mcstas import __all__  # noqa: F401


def __getattr__(name: str):
    """Return attributes from the replacement :mod:`ess.beer.mcstas` module."""
    return getattr(_mcstas, name)
