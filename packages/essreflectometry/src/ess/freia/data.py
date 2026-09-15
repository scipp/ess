# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from ess.reduce.data import make_registry

from ..reflectometry.types import Filename, ReferenceRun, SampleRun

_registry = make_registry(
    "ess/freia",
    version="1",
    files={},
)


def freia_mcstas_sample_run() -> Filename[SampleRun]:
    """Return path to the McStas sample events file."""
    return Filename[SampleRun](_registry.get_path("mcstas-sample.h5"))


def freia_mcstas_reference_run() -> Filename[ReferenceRun]:
    """Return path to the McStas reference events file."""
    return Filename[ReferenceRun](_registry.get_path("mcstas-reference.h5"))


__all__ = ["freia_mcstas_reference_run", "freia_mcstas_sample_run"]
