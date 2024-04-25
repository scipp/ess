# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from ..reflectometry.types import FilePath, PoochFilename, Run

_version = '1'

__all__ = ['get_path']


def _make_pooch():
    import pooch

    return pooch.create(
        path=pooch.os_cache('ess/amor'),
        env='ESS_AMOR_DATA_DIR',
        base_url='https://public.esss.dk/groups/scipp/ess/amor/{version}/',
        version=_version,
        registry={
            "reference.nxs": "md5:56d493c8051e1c5c86fb7a95f8ec643b",
            "sample.nxs": "md5:4e07ccc87b5c6549e190bc372c298e83",
            "amor2023n000608.hdf": "md5:e3a8b5b8495bb9ab2173848096df49d6",
            "amor2023n000609.hdf": "md5:d899d65f684cade2905f203f7d0fb326",
            "amor2023n000610.hdf": "md5:c9367d49079edcd17fa0b98e33326b05",
            "amor2023n000611.hdf": "md5:9da41177269faac0d936806393427837",
            "amor2023n000612.hdf": "md5:602f1bfcdbc1f618133c93a117d05f12",
            "amor2023n000613.hdf": "md5:ba0fbcbf0b45756a269eb3e943371ced",
            "amor2023n000614.hdf": "md5:18e8a755d6fd671758fe726de058e707",
        },
    )


_pooch = _make_pooch()


def get_path(filename: PoochFilename[Run]) -> FilePath[Run]:
    """
    Return the path to a data file bundled with scippneutron.

    This function only works with example data and cannot handle
    paths to custom files.
    """
    return FilePath[Run](_pooch.fetch(filename))


providers = (get_path,)
