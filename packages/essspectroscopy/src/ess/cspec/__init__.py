# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""CSPEC data reduction."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("essspectroscopy")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

__all__ = []
