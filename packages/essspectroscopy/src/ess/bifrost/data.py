# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Data for tests and documentation with BIFROST."""

_version = "2"


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache("ess/bifrost"),
        env="ESS_DATA_DIR",
        base_url="https://public.esss.dk/groups/scipp/ess/bifrost/{version}/",
        version=_version,
        registry={
            "BIFROST_20240914T053723.h5": "md5:0f2fa5c9a851f8e3a4fa61defaa3752e",
            "computed_energy_data_simulated.h5": "md5:31b2444042935e99831c3c84a375ccae",
            "BIFROST-simulation-tof-lookup-table.h5": "md5:f2c12eb39d0750238805d0e5a5d561f7",  # noqa: E501
        },
    )


_pooch = _make_pooch()


def get_path(name: str) -> str:
    """
    Return the path to a data file bundled with ess.bifrost.

    This function only works with example data and cannot handle
    paths to custom files.
    """
    return _pooch.fetch(name)


def simulated_elastic_incoherent_with_phonon() -> str:
    """Simulated data for elastic incoherent scattering including a phonon."""
    return get_path("BIFROST_20240914T053723.h5")


def tof_lookup_table_simulation() -> str:
    """Time-of-flight lookup table for the simulated BIFROST data."""
    return get_path("BIFROST-simulation-tof-lookup-table.h5")


def computed_energy_data_simulated() -> str:
    """Energy and momentum transfer computed from the simulated BIFROST data."""
    return get_path("computed_energy_data_simulated.h5")


__all__ = [
    "computed_energy_data_simulated",
    "get_path",
    "simulated_elastic_incoherent_with_phonon",
    "tof_lookup_table_simulation",
]
