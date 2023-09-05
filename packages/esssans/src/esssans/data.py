# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
_version = '1'

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
            'SANS2D00063091.hdf5': 'md5:c212f1c8c68db69eae88eca90a19e7e6',
            'SANS2D00063114.hdf5': 'md5:806a5780ff02676afcea1c3d8777ee21',
            'SANS2D00063159.hdf5': 'md5:7be098bcc1f4ca73394584076a99146d',
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
