# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import importlib.metadata

import lazy_loader as lazy

try:
    __version__ = importlib.metadata.version("essreduce")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[
        "nexus",
        "normalization",
        "polarization",
        "streaming",
        "uncertainty",
        "unwrap",
    ],
)
