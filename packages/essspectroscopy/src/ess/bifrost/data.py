# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Data for tests and documentation with BIFROST."""

from pathlib import Path

from ess.reduce.data import make_registry

_registry = make_registry(
    'ess/bifrost',
    files={
        "BIFROST_20240914T053723.h5": "md5:0f2fa5c9a851f8e3a4fa61defaa3752e",
        "computed_energy_data_simulated.h5": "blake2b:690d837ce684a0aeea021b2624c5f7371e299f1bbc803725b40bf3eca35816e3348a79e5fdf04dc89e29c4bdd4f6ea2b052b49058eecfdf535b9733eefa8a854",  # noqa: E501
        "computed_energy_data_simulated_5x2.h5": "blake2b:0353f7e675a276451637fd175eef4f3d231a68a8983faf1240796fef44c9f3e78a264baf705e24d71f001afc83d07bbf1bff17e5bd83a15db89386a43dd5d5d7",  # noqa: E501
        "BIFROST-simulation-tof-lookup-table.h5": "blake2b:682021920a355f789da37b18029719fe20569d86db26cdaf5f3d916d2f76f9360907960ba86903be4cab489d39f1b6f9f265f3a4ab3f82c5e095afa4a2c456af",  # noqa: E501
    },
    version="4",
)


def get_path(name: str) -> Path:
    """
    Return the path to a data file bundled with ess.bifrost.

    This function only works with example data and cannot handle
    paths to custom files.
    """
    return _registry.get_path(name)


def simulated_elastic_incoherent_with_phonon() -> Path:
    """Simulated data for elastic incoherent scattering including a phonon."""
    return get_path("BIFROST_20240914T053723.h5")


def tof_lookup_table_simulation() -> Path:
    """Time-of-flight lookup table for the simulated BIFROST data.

    This table was computed with `tof <https://github.com/scipp/tof>`_
    using `Create a time-of-flight lookup table for BIFROST
    <../../user-guide/bifrost/bifrost-make-tof-lookup-table.rst>`_
    with ``NumberOfSimulatedNeutrons = 5_000_000``.
    """
    return get_path("BIFROST-simulation-tof-lookup-table.h5")


def computed_energy_data_simulated() -> Path:
    """Energy and momentum transfer computed from the simulated BIFROST data."""
    return get_path("computed_energy_data_simulated.h5")


def computed_energy_data_simulated_5x2() -> Path:
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
