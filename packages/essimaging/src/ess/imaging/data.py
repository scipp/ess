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
            'siemens_star.tiff': 'md5:0ba27c2daf745338959f5156a3b0a2c0',
            'resolving_power_test_target.tiff': 'md5:a5d414603797f4cc02fe7b2ae4d7aa48',
            # Measurements that Søren Schmidt (imaging IDS 2025) made at JPark.
            "siemens-star-measured.h5": "md5:8e333d36c7c102f474b2b66cb785f5e8",
            "siemens-star-openbeam.h5": "md5:ee429b2c247aeaafb0ef3ca4171f2e6a",
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


def siemens_star_path() -> pathlib.Path:
    """
    Return the path to the Siemens star test image.
    """

    return get_path('siemens_star.tiff')


def resolving_power_test_target_path() -> pathlib.Path:
    """
    Return the path to the resolving power test target image.
    """

    return get_path('resolving_power_test_target.tiff')
