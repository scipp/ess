# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)


from ...data import Registry

_registry = Registry(
    instrument='powgen',
    files={
        'PG3_4844_event.nxs': 'md5:d5ae38871d0a09a28ae01f85d969de1e',
        'PG3_4866_event.nxs': 'md5:3d543bc6a646e622b3f4542bc3435e7e',
        'PG3_5226_event.nxs': 'md5:58b386ebdfeb728d34fd3ba00a2d4f1e',
        'PG3_FERNS_d4832_2011_08_24.cal': 'md5:c181221ebef9fcf30114954268c7a6b6',
    },
    version='1',
)

get_path = _registry.get_path


def sample_file() -> str:
    return get_path('PG3_4844_event.nxs')


def vanadium_file() -> str:
    return get_path('PG3_4866_event.nxs')


def empty_instrument_file() -> str:
    return get_path('PG3_5226_event.nxs')


def calibration_file() -> str:
    return get_path('PG3_FERNS_d4832_2011_08_24.cal')
