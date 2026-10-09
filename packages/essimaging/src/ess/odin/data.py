# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Data for tests and documentation with ODIN."""

import pathlib

from ess.reduce.data import make_registry

_registry = make_registry(
    'ess/odin',
    version="3",
    files={
        "iron_simulation_sample_small.nxs": "md5:9c94a683c2042bf2a79b183dbb01066b",
        "iron_simulation_ob_small.nxs": "md5:123494aa6c43f9a4584ba8454ccd2fd6",
        "iron_simulation_sample_large.nxs": "md5:4e133486837e3709885207d7cb1bd589",
        "iron_simulation_ob_large.nxs": "md5:69d5955dd0ca366e304b396523308699",
        "ODIN-wavelength-lookup-table-5m-65m.h5": "md5:44eef2a2e826cec688aeb1b985eb9f9e",  # noqa: E501
        "ymir_lego_odin.hdf": "md5:59b56b4ca2a264983df2d5590853c9fa",
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
