# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)


from ..data import Registry

_registry = Registry(
    instrument='loki',
    files={
        '60248-2022-02-28_2215.nxs': 'md5:d9f17b95274a0fc6468df7e39df5bf03',
        '60250-2022-02-28_2215.nxs': 'md5:6a519ceaacbae702a6d08241e86799b1',
        '60339-2022-02-28_2215.nxs': 'md5:03c86f6389566326bb0cbbd80b8f8c4f',
        '60392-2022-02-28_2215.nxs': 'md5:9ecc1a9a2c05a880144afb299fc11042',
        '60393-2022-02-28_2215.nxs': 'md5:bf550d0ba29931f11b7450144f658652',
        '60394-2022-02-28_2215.nxs': 'md5:c40f38a62337d86957af925296c4c615',
        'PolyGauss_I0-50_Rg-60.txt': 'md5:389ee172a139c06d062c26b5340ae9ce',
    },
    version='2',
)


get_path = _registry.get_path

__all__ = ['get_path']
