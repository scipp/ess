# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""Data for tests and documentation with BIFROST."""

_version = "1"

__all__ = ["get_path"]


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache("ess/bifrost"),
        env="ESS_DATA_DIR",
        base_url="https://public.esss.dk/groups/scipp/ess/bifrost/{version}/",
        version=_version,
        registry={
            "BIFROST_20240914T053723.h5": "md5:0f2fa5c9a851f8e3a4fa61defaa3752e",  # noqa: E501
        },
    )


_pooch = _make_pooch()


def get_path(name: str) -> str:
    """
    Return the path to a data file bundled with ess.biofrost.

    This function only works with example data and cannot handle
    paths to custom files.
    """
    return _pooch.fetch(name)


def simulated_elastic_incoherent_with_phonon() -> str:
    """Simulated data for elastic incoherent scattering including a phonon."""
    return get_path("BIFROST_20240914T053723.h5")
