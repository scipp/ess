# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pathlib

from ess.reduce.data import Entry, make_registry

_version = "0"

__all__ = ["get_path", "small_mcstas_2_sample", "small_mcstas_3_sample"]


_registry = make_registry(
    "ess/nmx",
    version="0",
    files={
        "small_mcstas_2_sample.h5": "md5:c3affe636397f8c9eea1d9c10a2bf487",
        "small_mcstas_3_sample.h5": "md5:2afaac205d13ee857ee5364e3f1957a7",
        "mtz_samples.tar.gz": Entry(
            alg="md5", chk="bed1eaf604bbe8725c1f6a20ca79fcc0", extractor="untar"
        ),
        "mtz_random_samples.tar.gz": Entry(
            alg="md5", chk="c8259ae2e605560ab88959e7109613b6", extractor="untar"
        ),
        "small_nmx_nexus.hdf": "md5:42cffb85e4ce7c1aaa5f7e81469b865e",
    },
)


def small_mcstas_2_sample() -> pathlib.Path:
    """McStas 2 file containing small number of events."""
    import warnings

    warnings.warn(
        DeprecationWarning(
            "``essnmx`` will not support loading files "
            "made by McStas with version less than 3 from ``25.0.0``. "
            "Use ``small_mcstas_3_sample`` instead."
        ),
        stacklevel=2,
    )

    return get_path("small_mcstas_2_sample.h5")


def small_mcstas_3_sample() -> pathlib.Path:
    """McStas 3 file that contains only ``bank0(1-3)`` in the ``data`` group.

    Real McStas 3 file should contain more dataset under ``data`` group.
    """
    return get_path("small_mcstas_3_sample.h5")


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

    return get_path("small_nmx_nexus.hdf")
