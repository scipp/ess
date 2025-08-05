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
        retry_if_failed=3,
        base_url="https://public.esss.dk/groups/scipp/ess/odin/{version}/",
        version=_version,
        registry={
            "iron_simulation_ob_large.nxs": "md5:a93517ea2aa167d134ca63671f663f99",
            "iron_simulation_ob_small.nxs": "md5:7591ed8f0adec2658fb08190bd530b12",
            "iron_simulation_sample_large.nxs": "md5:c162b6abeccb51984880d8d5002bae95",
            "iron_simulation_sample_small.nxs": "md5:dda6fb30aa88780c5a3d4cef6ea05278",
            "ODIN-tof-lookup-table.h5": "md5:e657021f4508f167b2a2eb550853b06b",
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
    The file was generated with the ``tools/mcstas_to_nexus.ipynb`` notebook, sampling
    1M events from the McStas results.
    """
    return get_path("iron_simulation_sample_small.nxs")


def iron_simulation_ob_small() -> str:
    """
    Thinned down version of McStas data stored in a Odin NeXus file with simulation
    of the open beam.
    The file was generated with the ``tools/mcstas_to_nexus.ipynb`` notebook, sampling
    1M events from the McStas results.
    """
    return get_path("iron_simulation_ob_small.nxs")


def iron_simulation_sample_large() -> str:
    """
    Full version of McStas data stored in a Odin NeXus file with simulation
    of an Fe sample.
    The file was generated with the ``tools/mcstas_to_nexus.ipynb`` notebook, sampling
    10M events from the McStas results.
    """
    return get_path("iron_simulation_sample_large.nxs")


def iron_simulation_ob_large() -> str:
    """
    Full version of McStas data stored in a Odin NeXus file with simulation
    of the open beam.
    The file was generated with the ``tools/mcstas_to_nexus.ipynb`` notebook, sampling
    10M events from the McStas results.
    """
    return get_path("iron_simulation_ob_large.nxs")


def odin_tof_lookup_table() -> str:
    """
    Odin TOF lookup table.
    This file is used to convert the time-of-flight to wavelength.

    This table was computed using `Create a time-of-flight lookup table for ODIN
    <../../user-guide/odin/odin-make-tof-lookup-table.rst>`_
    with ``NumberOfSimulatedNeutrons = 5_000_000``.
    """
    return get_path("ODIN-tof-lookup-table.h5")
