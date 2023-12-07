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
            'SANS2D00063091.hdf5': 'md5:1fdbe36a496e914336f2f9b5cad9f00e',
            'SANS2D00063114.hdf5': 'md5:536303077b9f55286af5ef6ec5124e1c',
            'SANS2D00063159.hdf5': 'md5:e2f2aea3ed9251e579210718de568203',
            '60248-2022-02-28_2215.nxs': 'md5:d9f17b95274a0fc6468df7e39df5bf03',
            '60250-2022-02-28_2215.nxs': 'md5:6a519ceaacbae702a6d08241e86799b1',
            '60339-2022-02-28_2215.nxs': 'md5:03c86f6389566326bb0cbbd80b8f8c4f',
            '60392-2022-02-28_2215.nxs': 'md5:9ecc1a9a2c05a880144afb299fc11042',
            '60393-2022-02-28_2215.nxs': 'md5:bf550d0ba29931f11b7450144f658652',
            '60394-2022-02-28_2215.nxs': 'md5:c40f38a62337d86957af925296c4c615',
            'PolyGauss_I0-50_Rg-60.txt': 'md5:389ee172a139c06d062c26b5340ae9ce',
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
