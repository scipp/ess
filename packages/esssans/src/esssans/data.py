# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
_version = '2'

__all__ = ['get_path']


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache('esssans'),
        env='ESSSANS_DATA_DIR',
        base_url='https://public.esss.dk/groups/scipp/ess/loki/{version}/',
        version=_version,
        registry={
            'DIRECT_SANS2D_REAR_34327_4m_8mm_16Feb16.hdf5': 'md5:43f4188301d709aa49df0631d03a67cb',  # noqa: E501
            'SANS2D00063091.nxs': 'md5:05929753ea06eca5fe4be164cb06b4d6',
            'SANS2D00063091.hdf5': 'md5:1fdbe36a496e914336f2f9b5cad9f00e',
            'SANS2D00063114.hdf5': 'md5:536303077b9f55286af5ef6ec5124e1c',
            'SANS2D00063159.hdf5': 'md5:e2f2aea3ed9251e579210718de568203',
            'SANS2D00063091.SolidAngle_from_mantid.hdf5': 'md5:d57b82db377cb1aea0beac7202713861',  # noqa: E501
        },
    )


_pooch = _make_pooch()


def get_path(name: str) -> str:
    """
    Return the path to a data file.

    This function only works with example data and cannot handle
    paths to custom files.
    """
    return _pooch.fetch(name)
