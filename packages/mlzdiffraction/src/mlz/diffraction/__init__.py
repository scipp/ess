# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Diffraction data reduction for MLZ instruments."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from . import powder, powtex, texture

try:
    __version__ = _version("mlzdiffraction")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = ["powder", "powtex", "texture"]
