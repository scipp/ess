# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import importlib.metadata

from . import nexus, normalization, uncertainty, unwrap

try:
    __version__ = importlib.metadata.version("essreduce")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

__all__ = ["nexus", "normalization", "uncertainty", "unwrap"]
