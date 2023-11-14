# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)


from ...data import Registry

_registry = Registry(
    instrument='powgen',
    files={
        # Files loadable with Mantid
        'PG3_4844_event.nxs': 'md5:d5ae38871d0a09a28ae01f85d969de1e',
        'PG3_4866_event.nxs': 'md5:3d543bc6a646e622b3f4542bc3435e7e',
        'PG3_5226_event.nxs': 'md5:58b386ebdfeb728d34fd3ba00a2d4f1e',
        'PG3_FERNS_d4832_2011_08_24.cal': 'md5:c181221ebef9fcf30114954268c7a6b6',
        # Zipped Scipp HDF5 files
        'PG3_4844_event.zip': 'md5:a644c74f5e740385469b67431b690a3e',
        'PG3_4866_event.zip': 'md5:5bc49def987f0faeb212a406b92b548e',
        'PG3_FERNS_d4832_2011_08_24.zip': 'md5:0fef4ed5f61465eaaa3f87a18f5bb80d',
    },
    version='1',
)

get_path = _registry.get_path


def mantid_sample_file() -> str:
    return get_path('PG3_4844_event.nxs')


def mantid_vanadium_file() -> str:
    return get_path('PG3_4866_event.nxs')


def mantid_empty_instrument_file() -> str:
    return get_path('PG3_5226_event.nxs')


def mantid_calibration_file() -> str:
    return get_path('PG3_FERNS_d4832_2011_08_24.cal')


def sample_file() -> str:
    print(get_path('PG3_4844_event.zip', unzip=True))
    (path,) = get_path('PG3_4844_event.zip', unzip=True)
    return path


def vanadium_file() -> str:
    print(get_path('PG3_4866_event.zip', unzip=True))
    (path,) = get_path('PG3_4866_event.zip', unzip=True)
    return path


def calibration_file() -> str:
    print(get_path('PG3_FERNS_d4832_2011_08_24.zip', unzip=True))
    (path,) = get_path('PG3_FERNS_d4832_2011_08_24.zip', unzip=True)
    return path
