# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry.types import Filename, ReferenceRun, SampleRun

_version = "1"


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache("ess/estia"),
        env="ESS_ESTIA_DATA_DIR",
        base_url="https://public.esss.dk/groups/scipp/ess/estia/{version}/",
        version=_version,
        registry={
            "218610_tof_detector_list.p.x.y.t.L.sx.sy": "md5:65145a26c36d12954a97d27d6e7f4ed9",  # noqa: E501
            "218611_tof_detector_list.p.x.y.t.L.sx.sy": "md5:4599e938568f3b73a72d6d48fe5160e7",  # noqa: E501
            "218612_tof_detector_list.p.x.y.t.L.sx.sy": "md5:6bacd1e4d922007c7f574f20378b28f2",  # noqa: E501
            "218613_tof_detector_list.p.x.y.t.L.sx.sy": "md5:7c17cb8a2fe38f4f0976de1254295636",  # noqa: E501
            "218614_tof_detector_list.p.x.y.t.L.sx.sy": "md5:78cf399dcedea2a2d4178e11b95c53f2",  # noqa: E501
            # McStas runs for the samples in the Estia McStas model at various angles
            "examples/220573/mccode.h5": "md5:2f1ba298bd8a1a67082a41d2c3957fc7",
            "examples/220574/mccode.h5": "md5:ecfcf72e936dd5345c1aededa7e9388b",
            "examples/220575/mccode.h5": "md5:efb206cf7cf87bf587bca31dca5da39d",
            "examples/220576/mccode.h5": "md5:f63664809a15331d055ab9746f7f7259",
            "examples/220577/mccode.h5": "md5:e0915c1271ae1800deba7e8c4b7cbf3e",
            "examples/220578/mccode.h5": "md5:c456c1bf7e30c09d3ab36f0a87721d49",
            "examples/220579/mccode.h5": "md5:3fd9ce4eab6c54346ce7086055695552",
            "examples/220580/mccode.h5": "md5:36ff5b178fef6eb5f3be97089c493574",
            "examples/220581/mccode.h5": "md5:736aa90d3570a104ab1ca0f4090ea94e",
            "examples/220582/mccode.h5": "md5:d01a1fa2e98f0086b3bc0e5dbc323bde",
            "examples/220583/mccode.h5": "md5:4f243df1b2dac2cf6da408ee6afb9a37",
            "examples/220584/mccode.h5": "md5:5d82b18156c3675a986ec2c02d96a45c",
            "examples/220585/mccode.h5": "md5:09540bd8ccea696232b1fdaac978cb29",
            # Ground truth reflectivity curves
            "examples/NiTiML.ref": "md5:9769b884dfa09d34b6ae449b640463a1",
            "examples/Si-SiO2.ref": "md5:436e2312e137b63bf31cc39064d28864",
            "examples/Si-Ni.ref": "md5:76fcfc655635086060387163c9175ab4",
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


def estia_mcstas_example(name):
    """Returns a list of McStas files associated with the sample."""
    if name == 'reference':
        return _pooch.fetch("examples/220573/mccode.h5")
    elif name == 'Ni/Ti-multilayer':
        return list(
            map(_pooch.fetch, [f"examples/2205{i}/mccode.h5" for i in range(74, 78)])
        )
    elif name == 'Ni-film on silicon':
        return list(
            map(_pooch.fetch, [f"examples/2205{i}/mccode.h5" for i in range(78, 82)])
        )
    elif name == 'Natural SiO2 on silicon':
        return list(
            map(_pooch.fetch, [f"examples/2205{i}/mccode.h5" for i in range(82, 86)])
        )
    raise ValueError(f'"{name}" is not a valid sample name')


def estia_mcstas_groundtruth(name):
    """Returns the ground truth reflectivity curve for the sample."""

    def parse(fname):
        ds = sc.io.load_csv(fname, sep=' ')
        if len(ds.keys()) == 3:
            _, Qcol, Rcol = ds.keys()
        else:
            Qcol, Rcol = ds.keys()
        return sc.DataArray(
            sc.array(dims='Q', values=ds[Rcol].values),
            coords={
                'Q': sc.array(dims='Q', values=[0, *ds[Qcol].values], unit='1/angstrom')
            },
        )

    if name == 'Ni/Ti-multilayer':
        return parse(_pooch.fetch("examples/NiTiML.ref"))
    elif name == 'Ni-film on silicon':
        return parse(_pooch.fetch("examples/Si-Ni.ref"))
    elif name == 'Natural SiO2 on silicon':
        return parse(_pooch.fetch("examples/Si-SiO2.ref"))
    raise ValueError(f'"{name}" is not a valid sample name')


__all__ = [
    "estia_mcstas_example",
    "estia_mcstas_reference_run",
    "estia_mcstas_sample_run",
]
