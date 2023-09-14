# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Tools for handling statistical uncertainties."""

from typing import Dict, TypeVar, Union, overload

import scipp as sc

T = TypeVar("T", bound=Union[sc.Variable, sc.DataArray])


@overload
def broadcast_with_upper_bound_variances(
    data: sc.Variable, sizes: Dict[str, int]
) -> sc.Variable:
    pass


@overload
def broadcast_with_upper_bound_variances(
    data: sc.DataArray, sizes: Dict[str, int]
) -> sc.DataArray:
    pass


def broadcast_with_upper_bound_variances(
    data: Union[sc.Variable, sc.DataArray], sizes: Dict[str, int]
) -> Union[sc.Variable, sc.DataArray]:
    if data.variances is None:
        return data
    if all(data.sizes.get(dim) == size for dim, size in sizes.items()):
        return data
    size = 1
    for dim, dim_size in sizes.items():
        if dim not in data.dims:
            size *= dim_size
    data = data.copy()
    data.variances = data.variances * size
    return data.broadcast(sizes={**sizes, **data.sizes}).copy()
