# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pathlib

import pooch

_version = "0"

__all__ = ["small_mcstas_2_sample", "small_mcstas_3_sample", "get_path"]


def _make_pooch() -> pooch.Pooch:
    return pooch.create(
        path=pooch.os_cache("essnmx"),
        env="ESSNMX_DATA_DIR",
        retry_if_failed=3,
        base_url="https://public.esss.dk/groups/scipp/ess/nmx/",
        version=_version,
        registry={
            "small_mcstas_2_sample.h5": "md5:c3affe636397f8c9eea1d9c10a2bf487",
            "small_mcstas_3_sample.h5": "md5:2afaac205d13ee857ee5364e3f1957a7",
            "mtz_samples.tar.gz": "md5:bed1eaf604bbe8725c1f6a20ca79fcc0",
            "mtz_random_samples.tar.gz": "md5:c8259ae2e605560ab88959e7109613b6",
        },
    )


_pooch = _make_pooch()


def small_mcstas_2_sample():
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


def small_mcstas_3_sample():
    """McStas 3 file that contains only ``bank0(1-3)`` in the ``data`` group.

    Real McStas 3 file should contain more dataset under ``data`` group.
    """
    return get_path("small_mcstas_3_sample.h5")


def get_path(name: str) -> str:
    """
    Return the path to a data file bundled with ess nmx.

    This function only works with example data and cannot handle
    paths to custom files.
    """
    return _pooch.fetch(name)


def get_small_mtz_samples() -> list[pathlib.Path]:
    """Return a list of path to MTZ sample files randomly chosen from real dataset.

    This samples also contain optional columns.
    """
    from pooch.processors import Untar

    return [
        pathlib.Path(file_path)
        for file_path in _pooch.fetch("mtz_samples.tar.gz", processor=Untar())
    ]


def get_small_random_mtz_samples() -> list[pathlib.Path]:
    """Return a list of path to MTZ sample files filled with random values

    This sample only contains mandatory columns for the workflow examples.
    They are made for documentation, not necessarily for testing.
    Use ``get_small_mtz_samples`` for testing since they are
    more representative of real data.
    """
    from pooch.processors import Untar

    return [
        pathlib.Path(file_path)
        for file_path in _pooch.fetch("mtz_random_samples.tar.gz", processor=Untar())
    ]
