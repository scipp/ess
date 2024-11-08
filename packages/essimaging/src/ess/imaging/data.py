# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pathlib

import pooch

_version = '1'


def _make_pooch():
    return pooch.create(
        path=pooch.os_cache('essimaging'),
        env='BEAMLIME_DATA_DIR',
        retry_if_failed=3,
        base_url=f'https://public.esss.dk/groups/scipp/ess/imaging/{_version}/',
        version=_version,
        registry={
            'small_mcstas_ob_images.h5': 'md5:2f181bbacb164c28bfaf7cce09701d92',
            'small_mcstas_sample_images.h5': 'md5:3c42570951cabec7caedc76d90d03fa3',
            'small_ymir_images.hdf': 'md5:cf83695d5da29e686c10a31b402b8bdb',
            'README.md': 'md5:9e1beeb325f127d691a8d7882db3255d',
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


def get_mcstas_ob_images_path() -> pathlib.Path:
    """
    Return the path to the small McStas OB images HDF5 file.
    """

    return get_path('small_mcstas_ob_images.h5')


def get_mcstas_sample_images_path() -> pathlib.Path:
    """
    Return the path to the small McStas sample images HDF5 file.
    """

    return get_path('small_mcstas_sample_images.h5')
