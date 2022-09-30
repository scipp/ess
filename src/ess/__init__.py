# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

# flake8: noqa: F401

try:
    from . import _version
    __version__ = _version.__version__
except ImportError:
    __version__ = "0.0.0-unknown"

from .logging import get_logger
from . import logging
