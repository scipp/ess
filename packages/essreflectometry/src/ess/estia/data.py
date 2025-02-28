# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from ..reflectometry.types import Filename, ReferenceRun, SampleRun

_version = "1"


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache("ess/amor"),
        env="ESS_ESTIA_DATA_DIR",
        base_url="https://public.esss.dk/groups/scipp/ess/estia/{version}/",
        version=_version,
        registry={
            "218610_tof_detector_list.p.x.y.t.L.sx.sy": "md5:65145a26c36d12954a97d27d6e7f4ed9",  # noqa: E501
            "218611_tof_detector_list.p.x.y.t.L.sx.sy": "md5:4599e938568f3b73a72d6d48fe5160e7",  # noqa: E501
            "218612_tof_detector_list.p.x.y.t.L.sx.sy": "md5:6bacd1e4d922007c7f574f20378b28f2",  # noqa: E501
            "218613_tof_detector_list.p.x.y.t.L.sx.sy": "md5:7c17cb8a2fe38f4f0976de1254295636",  # noqa: E501
            "218614_tof_detector_list.p.x.y.t.L.sx.sy": "md5:78cf399dcedea2a2d4178e11b95c53f2",  # noqa: E501
        },
    )


_pooch = _make_pooch()


def estia_mcstas_reference_run() -> Filename[ReferenceRun]:
    return Filename[ReferenceRun](
        _pooch.fetch("218610_tof_detector_list.p.x.y.t.L.sx.sy")
    )


def estia_mcstas_sample_run(number: int | str) -> Filename[SampleRun]:
    return Filename[SampleRun](
        _pooch.fetch(f"2186{int(number):02d}_tof_detector_list.p.x.y.t.L.sx.sy")
    )


__all__ = [
    "estia_mcstas_reference_run",
    "estia_mcstas_sample_run",
]
