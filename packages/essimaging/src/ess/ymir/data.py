# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pathlib

import pooch

_version = '1'


def _make_pooch():
    return pooch.create(
        path=pooch.os_cache('essimaging'),
        env='ESS_DATA_DIR',
        retry_if_failed=3,
        base_url=f'https://public.esss.dk/groups/scipp/ess/ymir/{_version}/',
        version=_version,
        registry={
            'small_ymir_images.hdf': 'md5:cf83695d5da29e686c10a31b402b8bdb',
        },
    )


_pooch = _make_pooch()


def get_path(name: str, unzip: bool = False) -> pathlib.Path:
    """
    Return the path to a data file bundled with ess.imaging.

    This function only works with example data and cannot handle
    paths to custom files.
    """
    if unzip:
        path = _pooch.fetch(name, processor=pooch.Unzip())[0]
    else:
        path = _pooch.fetch(name)

    return pathlib.Path(path)


def ymir_lego_images_path() -> pathlib.Path:
    """
    Return the path to the small YMIR images HDF5 file.
    """

    return get_path('small_ymir_images.hdf')
