# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
_version = '1'


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache('ess/offspec'),
        env='ESS_OFFSPEC_DATA_DIR',
        base_url='https://public.esss.dk/groups/scipp/ess/offspec/{version}/',
        version=_version,
        registry={
            "direct_beam.nxs": "md5:e929d3419b13c3ffa4a5545ec54f9044",
            "sample.nxs": "md5:f18a8122706201df8150e7556ae6eb59",
            "reduced_mantid.xye": "md5:1f372f51d2cefb8dee302cf0093b684f",
        },
    )


_pooch = _make_pooch()


def get_path(name: str) -> str:
    """
    Return the path to a data file bundled with scippneutron.

    This function only works with example data and cannot handle
    paths to custom files.
    """
    return _pooch.fetch(name)


def direct_beam() -> str:
    return get_path('direct_beam.nxs')


def sample() -> str:
    return get_path('sample.nxs')


def reduced_mantid() -> str:
    return get_path('reduced_mantid.xye')
