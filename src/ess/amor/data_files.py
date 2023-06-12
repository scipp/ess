# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)


from ..data import Registry

data_registry = Registry(
    instrument='amor',
    files={
        "reference.nxs": "md5:56d493c8051e1c5c86fb7a95f8ec643b",
        "sample.nxs": "md5:4e07ccc87b5c6549e190bc372c298e83",
    },
    version='1',
)
