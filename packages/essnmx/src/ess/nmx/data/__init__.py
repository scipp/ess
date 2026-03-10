# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pathlib

from ess.reduce.data import Entry, make_registry

_version = "1"

__all__ = [
    "get_path",
    "get_small_mcstas",
    "get_small_mtz_samples",
    "get_small_nmx_nexus",
    "get_small_random_mtz_samples",
]


_registry = make_registry(
    "ess/nmx",
    version=_version,
    files={
        "small_mcstas_sample.h5": "md5:2afaac205d13ee857ee5364e3f1957a7",
        "mtz_samples.tar.gz": Entry(
            alg="md5", chk="bed1eaf604bbe8725c1f6a20ca79fcc0", extractor="untar"
        ),
        "mtz_random_samples.tar.gz": Entry(
            alg="md5", chk="c8259ae2e605560ab88959e7109613b6", extractor="untar"
        ),
        "small_nmx_nexus.hdf.zip": Entry(
            alg="md5", chk="96877cddc9f6392c96890069657710ca", extractor="unzip"
        ),
    },
)


def get_small_mcstas() -> pathlib.Path:
    """McStas file that contains only ``bank0(1-3)`` in the ``data`` group.

    Real McStas file should contain more dataset under ``data`` group.
    McStas version >=3.
    """
    return get_path("small_mcstas_sample.h5")


def get_path(name: str) -> pathlib.Path:
    """
    Return the path to a data file bundled with ess nmx.

    This function only works with example data and cannot handle
    paths to custom files.
    """
    return _registry.get_path(name)


def get_small_mtz_samples() -> list[pathlib.Path]:
    """Return a list of path to MTZ sample files randomly chosen from real dataset.

    This samples also contain optional columns.
    """
    return _registry.get_paths("mtz_samples.tar.gz")


def get_small_random_mtz_samples() -> list[pathlib.Path]:
    """Return a list of path to MTZ sample files filled with random values

    This sample only contains mandatory columns for the workflow examples.
    They are made for documentation, not necessarily for testing.
    Use ``get_small_mtz_samples`` for testing since they are
    more representative of real data.
    """
    return _registry.get_paths("mtz_random_samples.tar.gz")


def get_small_nmx_nexus() -> pathlib.Path:
    """Return the path to a small NMX NeXus file."""

    return get_path("small_nmx_nexus.hdf.zip")
