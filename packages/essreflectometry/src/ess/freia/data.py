# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Data for tests and documentation with FREIA."""

from pathlib import Path

from ess.reduce.data import make_registry

from ..reflectometry.types import Filename, ReferenceRun, SampleRun

_registry = make_registry(
    "ess/freia",
    version="1",
    files={
        # WFM runs 265305 (silicon with natural oxide) and 265301 (no sample).
        "mcstas-wfm-silicon.h5": "md5:07892a4b705faa769f5be8d3601b529e",
        "mcstas-wfm-direct-beam.h5": "md5:b7a2f51615bdaec6957bf3af30751ebf",
        "Si-15SiO2-air.txt": "md5:2bac8babb5f7042fd2e3cbd62e22a278",
        "freia-wavelength-lookup-table-no-choppers-5m-35m.h5": "md5:e28793b7e1c12986ee63a1df68723268",  # noqa: E501
    },
)


def freia_mcstas_sample_run() -> Filename[SampleRun]:
    """Return the WFM McStas run for silicon with natural oxide."""
    return Filename[SampleRun](_registry.get_path("mcstas-wfm-silicon.h5"))


def freia_mcstas_reference_run() -> Filename[ReferenceRun]:
    """Return the matching WFM McStas direct-beam run without a sample."""
    return Filename[ReferenceRun](_registry.get_path("mcstas-wfm-direct-beam.h5"))


def freia_mcstas_silicon_reflectivity() -> Path:
    """Return the silicon reflectivity table used in the McStas sample run."""
    return _registry.get_path("Si-15SiO2-air.txt")


def freia_wavelength_lookup_table_no_choppers() -> Path:
    """Return the FREIA wavelength lookup table without choppers."""
    return _registry.get_path("freia-wavelength-lookup-table-no-choppers-5m-35m.h5")


__all__ = [
    "freia_mcstas_reference_run",
    "freia_mcstas_sample_run",
    "freia_mcstas_silicon_reflectivity",
    "freia_wavelength_lookup_table_no_choppers",
]
