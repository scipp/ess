# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
_version = '1'

__all__ = ['get_path']


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache('ess/dream'),
        env='ESS_DREAM_DATA_DIR',
        base_url='https://public.esss.dk/groups/scipp/ess/dream/{version}/',
        version=_version,
        registry={
            'data_dream_with_sectors.csv.zip': 'md5:52ae6eb3705e5e54306a001bc0ae85d8',
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
