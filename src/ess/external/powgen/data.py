# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
_version = '1'

__all__ = ['get_path']


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache('ess/powgen'),
        env='ESS_DATA_DIR',
        base_url='https://public.esss.dk/groups/scipp/ess/powgen/{version}/',
        version=_version,
        registry={
            'PG3_4844_event.nxs': 'md5:d5ae38871d0a09a28ae01f85d969de1e',
            'PG3_4866_event.nxs': 'md5:3d543bc6a646e622b3f4542bc3435e7e',
            'PG3_5226_event.nxs': 'md5:58b386ebdfeb728d34fd3ba00a2d4f1e',
            'PG3_FERNS_d4832_2011_08_24.cal': 'md5:c181221ebef9fcf30114954268c7a6b6',
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


def sample_file() -> str:
    return get_path('PG3_4844_event.nxs')


def vanadium_file() -> str:
    return get_path('PG3_4866_event.nxs')


def empty_instrument_file() -> str:
    return get_path('PG3_5226_event.nxs')


def calibration_file() -> str:
    return get_path('PG3_FERNS_d4832_2011_08_24.cal')
