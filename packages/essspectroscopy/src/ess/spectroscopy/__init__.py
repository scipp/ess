# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E402, F401, I

"""Data reduction for ESS spectroscopy experiments.

This package contains common functionality for spectrometers at ESS.
For instrument-specifics, see

- :mod:`ess.bifrost`
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("essspectroscopy")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib
