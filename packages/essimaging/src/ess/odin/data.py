# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Data for tests and documentation with ODIN."""

import pathlib

from ess.reduce.data import make_registry

_registry = make_registry(
    'ess/odin',
    version="3",
    files={
        "iron_simulation_sample_small.nxs": "md5:dc4c504844501453c55e65e8d211bef9",
        "iron_simulation_ob_small.nxs": "md5:823abcc264cbe60e532520f411b64148",
        "iron_simulation_sample_large.nxs": "md5:38b6eafef238ebe24d0a0e2b20374ec5",
        "iron_simulation_ob_large.nxs": "md5:b6e660f8b92e327e021c9112270f50cb",
        "ODIN-wavelength-lookup-table-5m-65m.h5": "md5:44eef2a2e826cec688aeb1b985eb9f9e",  # noqa: E501
        "ymir_lego_odin.hdf": "md5:8e8708891e2574046b6f372e5e3516a5",
    },
)


def iron_simulation_sample_small() -> pathlib.Path:
    """
    Thinned down version of McStas data stored in a Odin NeXus file with simulation
    of an Fe sample.
    The file was generated with the ``tools/mcstas_to_nexus.ipynb`` notebook, sampling
    1M events from the McStas results.
    """
    return _registry.get_path("iron_simulation_sample_small.nxs")


def iron_simulation_ob_small() -> pathlib.Path:
    """
    Thinned down version of McStas data stored in a Odin NeXus file with simulation
    of the open beam.
    The file was generated with the ``tools/mcstas_to_nexus.ipynb`` notebook, sampling
    1M events from the McStas results.
    """
    return _registry.get_path("iron_simulation_ob_small.nxs")


def iron_simulation_sample_large() -> pathlib.Path:
    """
    Full version of McStas data stored in a Odin NeXus file with simulation
    of an Fe sample.
    The file was generated with the ``tools/mcstas_to_nexus.ipynb`` notebook, sampling
    10M events from the McStas results.
    """
    return _registry.get_path("iron_simulation_sample_large.nxs")


def iron_simulation_ob_large() -> pathlib.Path:
    """
    Full version of McStas data stored in a Odin NeXus file with simulation
    of the open beam.
    The file was generated with the ``tools/mcstas_to_nexus.ipynb`` notebook, sampling
    10M events from the McStas results.
    """
    return _registry.get_path("iron_simulation_ob_large.nxs")


def odin_wavelength_lookup_table() -> pathlib.Path:
    """
    Odin wavelength lookup table.
    This file is used to convert the raw ``event_time_offset`` to wavelength.

    This table was computed using `Create a wavelength lookup table for ODIN
    <../../odin/odin-make-wavelength-lookup-table.rst>`_
    with ``NumberOfSimulatedNeutrons = 5_000_000``.
    """
    return _registry.get_path("ODIN-wavelength-lookup-table-5m-65m.h5")


def odin_lego_images() -> pathlib.Path:
    """
    Return the path to the ODIN LEGO HDF5 file, created from the YMIR data.
    This file was created using the tools/make-odin-images-from-ymir.ipynb notebook.
    A ODIN file (coda_odin_999999_00011093.hdf) was used as a template for the NeXus
    structure. The images were extracted from the YMIR LEGO run.
    """
    return _registry.get_path("ymir_lego_odin.hdf")
