# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Data for tests and documentation with ODIN."""

_version = "1"

__all__ = ["get_path"]


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache("ess/odin"),
        env="ESS_DATA_DIR",
        base_url="https://public.esss.dk/groups/scipp/ess/odin/{version}/",
        version=_version,
        registry={
            "small_mcstas_ob_images.nxs": "md5:2d0c85fbd1917d7616a6c2c408e09ac3",
            "small_mcstas_sample_images.nxs": "md5:e2deec45e04c05931b26713f184190c5",
        },
    )


_pooch = _make_pooch()


def get_path(name: str, unzip: bool = False) -> str:
    """
    Return the path to a data file bundled with ess.dream.

    This function only works with example data and cannot handle
    paths to custom files.
    """
    import pooch

    return _pooch.fetch(name, processor=pooch.Unzip() if unzip else None)


def small_mcstas_sample_images() -> str:
    """
    Thinned down version of McStas data stored in a Odin NeXus file with simulation
    of an Fe sample.
    """
    return get_path("small_mcstas_sample_images.nxs")


def small_mcstas_ob_images() -> str:
    """
    Thinned down version of McStas data stored in a Odin NeXus file with simulation
    of the open beam.
    """
    return get_path("small_mcstas_ob_images.nxs")
