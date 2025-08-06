# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Data for tests and documentation with TBL."""

_version = "1"

__all__ = ["get_path"]


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache("ess/tbl"),
        env="ESS_DATA_DIR",
        retry_if_failed=3,
        base_url="https://public.esss.dk/groups/scipp/ess/tbl/{version}/",
        version=_version,
        registry={
            "tbl_sample_data_2025-03.hdf": "md5:12db6bc06721278b3abe47992eac3e77",
            "TBL-tof-lookup-table-no-choppers.h5": "md5:8bc98fac0ee64fc8f5decf509c75bafe",  # noqa: E501
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


def tutorial_sample_data() -> str:
    """ """
    return get_path("tbl_sample_data_2025-03.hdf")


def tbl_tof_lookup_table_no_choppers() -> str:
    """
    TBL TOF lookup table without choppers.
    This file is used to convert the neutron arrival time to time-of-flight.

    This table was computed using `Create a time-of-flight lookup table for TBL
    <../../user-guide/tbl/tbl-make-tof-lookup-table.rst>`_
    with ``NumberOfSimulatedNeutrons = 2_000_000``.
    """
    return get_path("TBL-tof-lookup-table-no-choppers.h5")
