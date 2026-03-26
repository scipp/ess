# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from multiprocessing.pool import ThreadPool
from pathlib import Path

import scipp as sc

from ess.reduce.data import make_registry

from ..reflectometry.types import Filename, ReferenceRun, SampleRun

_registry = make_registry(
    "ess/estia",
    version="2",
    files={
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
        # McStas runs converted to NeXus files using event sampling
        "examples/220573.nx": "md5:8d40da6f860e63784cb2ea11690b52f6",
        "examples/220574.nx": "md5:146cbc92c8a749e584be2a9acc8de54f",
        "examples/220575.nx": "md5:ce6532c329b739916c3aaa269694ac7b",
        "examples/220576.nx": "md5:5299660f41e331987f6ed2a88f1cf243",
        "examples/220577.nx": "md5:1205ad3335e709dd44979090cb85944e",
        # Spin flip example from McStas simulation.
        # All runs have the same sample rotation angle but different samples.
        # Each sample is measured using all four flipper setting.
        "spin_flip_example/ground_truth_spin_down_reflectivity.h5": "md5:1b3f4c70be6e2d5bae35836c378a0762",  # noqa: E501
        "spin_flip_example/ground_truth_spin_up_reflectivity.h5": "md5:77ded33407a5004587e475ede0312424",  # noqa: E501
        "spin_flip_example/magnetic_supermirror_2_offoff.h5": "md5:79967f157e9ea0c6a37c6e2d0851d099",  # noqa: E501
        "spin_flip_example/magnetic_supermirror_2_offon.h5": "md5:8ad8bc6a72bbc65367f367ffd25847d7",  # noqa: E501
        "spin_flip_example/magnetic_supermirror_2_onoff.h5": "md5:1886ee2ab950e186122318b1d39309ca",  # noqa: E501
        "spin_flip_example/magnetic_supermirror_2_onon.h5": "md5:cf37b249c2c1821b14601df2fe9897b8",  # noqa: E501
        "spin_flip_example/magnetic_supermirror_offoff.h5": "md5:7b209e1d0ae00bfba956b17e261aab78",  # noqa: E501
        "spin_flip_example/magnetic_supermirror_offon.h5": "md5:288bfc8c355e4f2b71815cb71b928685",  # noqa: E501
        "spin_flip_example/magnetic_supermirror_onoff.h5": "md5:357f31b745d04f61bc5c12b5f138485d",  # noqa: E501
        "spin_flip_example/magnetic_supermirror_onon.h5": "md5:eadb542686850159c3a18818efd276b3",  # noqa: E501
        "spin_flip_example/spin_flip_sample_offoff.h5": "md5:6a9f7d2a2ffe7532873c4bc571e785d4",  # noqa: E501
        "spin_flip_example/spin_flip_sample_offon.h5": "md5:7aedce342969c7a6742db0ff291cfd1d",  # noqa: E501
        "spin_flip_example/spin_flip_sample_onoff.h5": "md5:33193aa189ac56dac00d3d9dbf0a5f9e",  # noqa: E501
        "spin_flip_example/spin_flip_sample_onon.h5": "md5:2a56601aceb4d9d90951ad290952bc11",  # noqa: E501
        "spin_flip_example/supermirror_offoff.h5": "md5:5774d0bd58a153b3b6322a64bf4bf805",  # noqa: E501
        "spin_flip_example/supermirror_offon.h5": "md5:68ea97d4c83410360a04cc55da4d0a33",  # noqa: E501
        "spin_flip_example/supermirror_onoff.h5": "md5:4e682089541046bf6c167c94068569e1",  # noqa: E501
        "spin_flip_example/supermirror_onon.h5": "md5:6fc2090e6947949fe937fe0bf42600ec",
        "estia-tof-lookup-table-pulse-stride-1.h5": "md5:696953c772209fadc9c4a2ca619cf04d",  # noqa: E501
        "estia-lookup-table-pulse-stride-1.h5": "md5:01787012c678ea6ddce6a5f3a33d8895",
    },
)


def estia_mcstas_reference_run() -> Filename[ReferenceRun]:
    """Return path to the McStas reference events file."""
    return Filename[ReferenceRun](
        _registry.get_path("218610_tof_detector_list.p.x.y.t.L.sx.sy")
    )


def estia_mcstas_sample_run(number: int | str) -> Filename[SampleRun]:
    """Return path to a McStas sample events file by run number."""
    return Filename[SampleRun](
        _registry.get_path(f"2186{int(number):02d}_tof_detector_list.p.x.y.t.L.sx.sy")
    )


def estia_mcstas_nexus_reference_example() -> Path:
    """Return path to a NeXus reference example file."""
    return _registry.get_path("examples/220573.nx")


def estia_mcstas_nexus_sample_example(name: str) -> list[Path]:
    """Returns a list of NeXus files created from CODA files
    filled with sampled McStas events."""
    if name == 'Ni/Ti-multilayer':
        return list(
            map(
                _registry.get_path,
                [f"examples/2205{i}.nx" for i in range(74, 78)],
            )
        )
    raise ValueError(f'"{name}" is not a valid sample name')


def estia_mcstas_reference_example() -> Path:
    """Return path to a McStas reference example file."""
    return _registry.get_path("examples/220573/mccode.h5")


def estia_mcstas_sample_example(name: str) -> list[Path]:
    """Returns a list of McStas files associated with the sample."""
    if name == 'Ni/Ti-multilayer':
        return list(
            map(
                _registry.get_path,
                [f"examples/2205{i}/mccode.h5" for i in range(74, 78)],
            )
        )
    elif name == 'Ni-film on silicon':
        return list(
            map(
                _registry.get_path,
                [f"examples/2205{i}/mccode.h5" for i in range(78, 82)],
            )
        )
    elif name == 'Natural SiO2 on silicon':
        return list(
            map(
                _registry.get_path,
                [f"examples/2205{i}/mccode.h5" for i in range(82, 86)],
            )
        )
    raise ValueError(f'"{name}" is not a valid sample name')


def estia_mcstas_groundtruth(name) -> sc.DataArray:
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
        return parse(_registry.get_path("examples/NiTiML.ref"))
    elif name == 'Ni-film on silicon':
        return parse(_registry.get_path("examples/Si-Ni.ref"))
    elif name == 'Natural SiO2 on silicon':
        return parse(_registry.get_path("examples/Si-SiO2.ref"))
    raise ValueError(f'"{name}" is not a valid sample name')


def estia_mcstas_spin_flip_example(sample, flipper_setting) -> Path:
    """Return path to a spin-flip McStas example file."""
    return _registry.get_path(f'spin_flip_example/{sample}_{flipper_setting}.h5')


def estia_mcstas_spin_flip_example_groundtruth(up_or_down) -> Path:
    """Return path to the spin-flip ground truth reflectivity curve."""
    if up_or_down == 'down':
        return _registry.get_path(
            'spin_flip_example/ground_truth_spin_down_reflectivity.h5'
        )
    if up_or_down == 'up':
        return _registry.get_path(
            'spin_flip_example/ground_truth_spin_up_reflectivity.h5'
        )
    raise ValueError(f'Ground truth curve for spin state "{up_or_down}" does not exist')


def _refresh_cache(args):
    estia_mcstas_spin_flip_example(*args)


def estia_mcstas_spin_flip_example_download_all_to_cache() -> None:
    """Download all spin-flip example files into the local cache."""
    # Run once to create the folder structure without conflicts
    _refresh_cache(('supermirror', 'offoff'))
    with ThreadPool(20) as pool:
        for _ in pool.map(
            _refresh_cache,
            [
                (sample, setting)
                for sample in (
                    'supermirror',
                    'magnetic_supermirror',
                    'magnetic_supermirror_2',
                    'spin_flip_sample',
                )
                for setting in ('offoff', 'offon', 'onoff', 'onon')
            ],
        ):
            pass


def estia_wavelength_lookup_table() -> Path:
    """Return path to the ESTIA wavelength lookup table."""
    return _registry.get_path('estia-lookup-table-pulse-stride-1.h5')


__all__ = [
    "estia_mcstas_nexus_reference_example",
    "estia_mcstas_nexus_sample_example",
    "estia_mcstas_reference_example",
    "estia_mcstas_reference_run",
    "estia_mcstas_sample_example",
    "estia_mcstas_sample_run",
    "estia_mcstas_spin_flip_example",
    "estia_mcstas_spin_flip_example_download_all_to_cache",
    "estia_mcstas_spin_flip_example_groundtruth",
]
