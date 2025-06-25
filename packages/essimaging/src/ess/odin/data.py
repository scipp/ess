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
            "iron_simulation_ob_large.nxs": "md5:a93517ea2aa167d134ca63671f663f99",
            "iron_simulation_ob_small.nxs": "md5:7591ed8f0adec2658fb08190bd530b12",
            "iron_simulation_sample_large.nxs": "md5:c162b6abeccb51984880d8d5002bae95",
            "iron_simulation_sample_small.nxs": "md5:dda6fb30aa88780c5a3d4cef6ea05278",
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


def iron_simulation_sample_small() -> str:
    """
    Thinned down version of McStas data stored in a Odin NeXus file with simulation
    of an Fe sample.
    """
    return get_path("iron_simulation_sample_small.nxs")


def iron_simulation_ob_small() -> str:
    """
    Thinned down version of McStas data stored in a Odin NeXus file with simulation
    of the open beam.
    """
    return get_path("iron_simulation_ob_small.nxs")


def iron_simulation_sample_large() -> str:
    """
    Full version of McStas data stored in a Odin NeXus file with simulation
    of an Fe sample.
    """
    return get_path("iron_simulation_sample_large.nxs")


def iron_simulation_ob_large() -> str:
    """
    Full version of McStas data stored in a Odin NeXus file with simulation
    of the open beam.
    """
    return get_path("iron_simulation_ob_large.nxs")
