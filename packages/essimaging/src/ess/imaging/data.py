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
        base_url=f'https://public.esss.dk/groups/scipp/ess/imaging/{_version}/',
        version=_version,
        registry={
            'small_mcstas_ob_images.h5': 'md5:2f181bbacb164c28bfaf7cce09701d92',
            'small_mcstas_sample_images.h5': 'md5:3c42570951cabec7caedc76d90d03fa3',
            'small_ymir_images.hdf': 'md5:cf83695d5da29e686c10a31b402b8bdb',
            'README.md': 'md5:9e1beeb325f127d691a8d7882db3255d',
            'siemens_star.tiff': 'md5:0ba27c2daf745338959f5156a3b0a2c0',
            'resolving_power_test_target.tiff': 'md5:a5d414603797f4cc02fe7b2ae4d7aa48',
            'tbl-orca-focussing.hdf.zip': 'md5:f365acd9ea45dd205c0b9398d163cfa4',
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


def get_siemens_star_path() -> pathlib.Path:
    """
    Return the path to the Siemens star test image.
    """

    return get_path('siemens_star.tiff')


def get_resolving_power_test_target_path() -> pathlib.Path:
    """
    Return the path to the resolving power test target image.
    """

    return get_path('resolving_power_test_target.tiff')


def get_tbl_orca_focussing_path() -> pathlib.Path:
    """
    Return the path to the TBL ORCA HDF5 file used for camera focussing.
    Note that the images in this file have been resized from 2048x2048 to 512x512
    to reduce the file size.
    """

    return get_path('tbl-orca-focussing.hdf.zip', unzip=True)
