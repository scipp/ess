# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pathlib

import pooch

_version = '0'


def _make_pooch():
    return pooch.create(
        path=pooch.os_cache('essimaging'),
        env='BEAMLIME_DATA_DIR',
        retry_if_failed=3,
        base_url='https://public.esss.dk/groups/scipp/ess/imaging/',
        version=_version,
        registry={
            'small_ymir_images.hdf': 'md5:50489ee86793c4de23072fe52878619f',
            'README.md': 'md5:51339b8a7a29458b2ab38daf4071dc8d',
        },
    )


_pooch = _make_pooch()
_pooch.fetch('README.md')


def get_path(name: str) -> pathlib.Path:
    """
    Return the path to a data file bundled with ess.imaging test helpers.

    This function only works with example data and cannot handle
    paths to custom files.
    """

    return pathlib.Path(_pooch.fetch(name))


def get_ymir_images_path() -> pathlib.Path:
    """
    Return the path to the small YMIR images HDF5 file.
    """

    return get_path('small_ymir_images.hdf')
