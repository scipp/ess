# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from ..data import Registry

_registry = Registry(
    instrument='dream',
    files={
        'data_dream_HF_mil_closed_alldets_1e9.csv.zip': 'md5:2997f33d3ddc792083bfab661cf1d93a'  # noqa: E501
    },
    version='1',
)

get_path = _registry.get_path
