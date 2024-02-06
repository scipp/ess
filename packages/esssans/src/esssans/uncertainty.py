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
    if _no_variance_broadcast(data, sizes):
        return data
    size = 1
    for dim, dim_size in sizes.items():
        if dim not in data.dims:
            size *= dim_size
    data = data.copy()
    data.variances *= size
    return data.broadcast(sizes={**sizes, **data.sizes}).copy()


def drop_variances_if_broadcast(
    data: Union[sc.Variable, sc.DataArray], sizes: Dict[str, int]
) -> Union[sc.Variable, sc.DataArray]:
    if _no_variance_broadcast(data, sizes):
        return data
    return sc.values(data)


def _no_variance_broadcast(
    data: Union[sc.Variable, sc.DataArray], sizes: Dict[str, int]
) -> bool:
    return (data.variances is None) or all(
        data.sizes.get(dim) == size for dim, size in sizes.items()
    )


def broadcast_to_events_with_upper_bound_variances(
    da: sc.DataArray, *, events: sc.DataArray
) -> sc.DataArray:
    """
    Upper-bound estimate for errors from normalization in event-mode.

    Count the number of events in each bin of the input data array. Then scale the
    variances by the number of events in each bin. An explicit broadcast is performed
    to bypass Scipp's safety check on broadcasting variances.

    Details will be published in an upcoming publication by Simon Heybrock et al.
    """
    if da.variances is None:
        return da
    constituents = events.bins.constituents

    if 'Q' in constituents['data'].coords:
        Q = constituents['data'].coords['Q']
        constituents['data'] = sc.DataArray(
            sc.ones(sizes=Q.sizes, dtype='float32'), coords={'Q': Q}
        )
        edges = {'Q': da.coords['Q']}
    else:
        Qx = constituents['data'].coords['Qx']
        Qy = constituents['data'].coords['Qy']
        constituents['data'] = sc.DataArray(
            sc.ones(sizes=Qx.sizes, dtype='float32'),
            coords={'Qx': Qx, 'Qy': Qy},
        )
        edges = {'Qy': da.coords['Qy'], 'Qx': da.coords['Qx']}
    # Combine all bins of the events that correspond to the same bin in the input data
    to_concat = set(events.dims) - set(da.dims)
    binned = sc.DataArray(sc.bins(**constituents).bins.concat(to_concat))
    counts = binned.hist(**edges)
    da = da.copy()
    da.variances *= counts.values
    da.data = sc.bins_like(events, da.data)
    return da
