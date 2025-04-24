# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from ..reflectometry.types import Filename, ReferenceRun, SampleRun

_version = "1"


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache("ess/offspec"),
        env="ESS_AMOR_DATA_DIR",
        base_url="https://public.esss.dk/groups/scipp/ess/offspec/{version}/",
        version=_version,
        registry={
            "sample.h5": "md5:02b8703230b6b1e6282c0d39eb94523c",
            "direct_beam.h5": "md5:1c4e56afbd35edd96c7607e357981ccf",
        },
    )


_pooch = _make_pooch()


def offspec_sample_run() -> Filename[SampleRun]:
    return Filename[SampleRun](_pooch.fetch("sample.h5"))


def offspec_direct_beam_run() -> Filename[ReferenceRun]:
    return Filename[ReferenceRun](_pooch.fetch("direct_beam.h5"))


__all__ = [
    "offspec_direct_beam_run",
    "offspec_sample_run",
]
