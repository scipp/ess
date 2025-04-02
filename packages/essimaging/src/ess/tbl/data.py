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
        base_url="https://public.esss.dk/groups/scipp/ess/tbl/{version}/",
        version=_version,
        registry={
            "tbl_sample_data_2025-03.hdf": "md5:12db6bc06721278b3abe47992eac3e77",
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
