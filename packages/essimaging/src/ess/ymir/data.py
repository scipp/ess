# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pathlib

from ..imaging.data import Registry

_registry = Registry(
    instrument='ymir',
    version="1",
    files={
        'small_ymir_images.hdf': 'md5:cf83695d5da29e686c10a31b402b8bdb',
    },
)


def ymir_lego_images_path() -> pathlib.Path:
    """
    Return the path to the small YMIR images HDF5 file.
    """

    return _registry('small_ymir_images.hdf')
