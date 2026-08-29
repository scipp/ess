# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Data for SKADI documentation examples."""

from pathlib import Path

from ess.reduce.data import make_registry

_registry = make_registry(
    "ess/skadi",
    files={
        "skadi_mcstas_1e8_sample10_1_of_50.h5": (
            "md5:37883335a05c41d420cff2b38f883fc2"
        ),
    },
    version="1",
)


def skadi_mcstas_sample() -> Path:
    """Return the reduced SKADI McStas sample used in the user guide."""
    return _registry.get_path("skadi_mcstas_1e8_sample10_1_of_50.h5")


__all__ = ["skadi_mcstas_sample"]
