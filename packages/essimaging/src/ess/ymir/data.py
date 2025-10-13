# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pathlib

from ess.reduce.data import make_registry

_registry = make_registry(
    'ess/ymir',
    version="1",
    files={
        'small_ymir_images.hdf': 'md5:cf83695d5da29e686c10a31b402b8bdb',
    },
)


def ymir_lego_images_path() -> pathlib.Path:
    """
    Return the path to the small YMIR images HDF5 file.
    """

    return _registry.get_path('small_ymir_images.hdf')
