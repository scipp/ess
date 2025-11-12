# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from ess.reduce.data import make_registry

from ..reflectometry.types import Filename, ReferenceRun, SampleRun

_registry = make_registry(
    "ess/offspec",
    version="1",
    files={
        "sample.h5": "md5:02b8703230b6b1e6282c0d39eb94523c",
        "direct_beam.h5": "md5:1c4e56afbd35edd96c7607e357981ccf",
    },
)


def offspec_sample_run() -> Filename[SampleRun]:
    return Filename[SampleRun](_registry.get_path("sample.h5"))


def offspec_direct_beam_run() -> Filename[ReferenceRun]:
    return Filename[ReferenceRun](_registry.get_path("direct_beam.h5"))


__all__ = [
    "offspec_direct_beam_run",
    "offspec_sample_run",
]
