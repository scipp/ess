# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pathlib

from ess.reduce.data import make_registry

_registry = make_registry(
    'ess/imaging',
    version="1",
    files={
        'siemens_star.tiff': 'md5:0ba27c2daf745338959f5156a3b0a2c0',
        'resolving_power_test_target.tiff': 'md5:a5d414603797f4cc02fe7b2ae4d7aa48',
        # Measurements that Søren Schmidt (imaging IDS 2025) made at J-PARC.
        "siemens-star-measured.h5": "md5:8e333d36c7c102f474b2b66cb785f5e8",
        "siemens-star-openbeam.h5": "md5:ee429b2c247aeaafb0ef3ca4171f2e6a",
    },
)


def siemens_star_path() -> pathlib.Path:
    """
    Return the path to the Siemens star test image.
    """

    return _registry.get_path('siemens_star.tiff')


def resolving_power_test_target_path() -> pathlib.Path:
    """
    Return the path to the resolving power test target image.
    """

    return _registry.get_path('resolving_power_test_target.tiff')


def jparc_siemens_star_measured_path() -> pathlib.Path:
    """
    Return the path to the Siemens star test image measured at J-PARC.
    """

    return _registry.get_path('siemens-star-measured.h5')


def jparc_siemens_star_openbeam_path() -> pathlib.Path:
    """
    Return the path to the Siemens star open beam image measured at J-PARC.
    """

    return _registry.get_path('siemens-star-openbeam.h5')
