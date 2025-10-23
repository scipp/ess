# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Data for tests and documentation with BIFROST."""

_version = "3"


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache("ess/bifrost"),
        env="ESS_DATA_DIR",
        base_url="https://public.esss.dk/groups/scipp/ess/bifrost/{version}/",
        version=_version,
        registry={
            "BIFROST_20240914T053723.h5": "md5:0f2fa5c9a851f8e3a4fa61defaa3752e",
            "computed_energy_data_simulated.h5": "blake2b:3c398443cb85c8294d283c7212255bc695c2520f2332c2c99d041a0760b6bcbb9937e19bcd8a498daf306d279c88d2ea911c510c1ce3b3a7f1e6b7e54022a943",  # noqa: E501
            "computed_energy_data_simulated_5x2.h5": "blake2b:d9d5e785a08e14d9c3416cf04db89a8c6f2fae3c0bae27bf0e73e8e5d492b4ca406e6578a935fa9f72dd9199dc15536409f614791fee6899c4265fe5d31e2706",  # noqa: E501
            "BIFROST-simulation-tof-lookup-table.h5": "blake2b:682021920a355f789da37b18029719fe20569d86db26cdaf5f3d916d2f76f9360907960ba86903be4cab489d39f1b6f9f265f3a4ab3f82c5e095afa4a2c456af",  # noqa: E501
        },
        retry_if_failed=3,
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
    """Time-of-flight lookup table for the simulated BIFROST data.

    This table was computed with `tof <https://github.com/scipp/tof>`_
    using `Create a time-of-flight lookup table for BIFROST
    <../../user-guide/bifrost/bifrost-make-tof-lookup-table.rst>`_
    with ``NumberOfSimulatedNeutrons = 5_000_000``.
    """
    return get_path("BIFROST-simulation-tof-lookup-table.h5")


def computed_energy_data_simulated() -> str:
    """Energy and momentum transfer computed from the simulated BIFROST data."""
    return get_path("computed_energy_data_simulated.h5")


def computed_energy_data_simulated_5x2() -> str:
    """Energy and momentum transfer computed from the simulated BIFROST data.

    This reference was computed with 10 detectors forming a 5x2 grid
    (arc=5, channel=2).
    """
    return get_path("computed_energy_data_simulated_5x2.h5")


__all__ = [
    "computed_energy_data_simulated",
    "computed_energy_data_simulated_5x2",
    "get_path",
    "simulated_elastic_incoherent_with_phonon",
    "tof_lookup_table_simulation",
]
