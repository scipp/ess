# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""Data for tests and documentation with BIFROST."""

from pathlib import Path

from ess.reduce.data import make_registry

_registry = make_registry(
    'ess/bifrost',
    files={
        "bifrost_260526T151446.repack.h5": "md5:0f60f5938b0dce87542dd525a2f9a5c4",
        "computed_energy_data_simulated_5x2.h5": "md5:2abdadbbcb48de02ebb37bd546fff534",
        "BIFROST-simulation-lookup-table.h5": "md5:0a4df8039b9610df80fb7bd567736eeb",
    },
    version="9",
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
    return get_path("bifrost_260526T151446.repack.h5")


def lookup_table_simulation() -> Path:
    """Wavelength lookup table for the simulated BIFROST data.

    This table was computed with `tof <https://github.com/scipp/tof>`_
    using `Create a wavelength lookup table for BIFROST
    <../../user-guide/bifrost/bifrost-make-wavelength-lookup-table.rst>`_
    with ``NumberOfSimulatedNeutrons = 5_000_000``.
    """
    return get_path("BIFROST-simulation-lookup-table.h5")


def computed_energy_data_simulated_5x2() -> Path:
    """Energy and momentum transfer computed from the simulated BIFROST data.

    This reference was computed with 10 detectors forming a 5x2 grid
    (arc=5, channel=2).
    """
    return get_path("computed_energy_data_simulated_5x2.h5")


__all__ = [
    "computed_energy_data_simulated_5x2",
    "get_path",
    "lookup_table_simulation",
    "simulated_elastic_incoherent_with_phonon",
]
