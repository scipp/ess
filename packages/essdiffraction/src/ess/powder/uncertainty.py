# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Tools for handling statistical uncertainties."""

from typing import Optional

import scipp as sc

from .types import RawData, RawDataWithVariances, UncertaintyBroadcastMode, VanadiumRun


# TODO remove
def drop_variances(data: RawDataWithVariances[VanadiumRun]) -> RawData[VanadiumRun]:
    res = data.copy(deep=False)
    if res.bins is not None:
        res.bins.constituents['data'].variances = None
    else:
        res.variances = None
    return RawData[VanadiumRun](res)


def broadcast_uncertainties(
    data: sc.DataArray, uncertainty_broadcast_mode: Optional[UncertaintyBroadcastMode]
) -> sc.DataArray:
    """Broadcast uncertainties using the specified mode.

    Parameters
    ----------
    data:
        Data with uncertainties to broadcast.
    uncertainty_broadcast_mode:
        Selected broadcast mode.

    Returns
    -------
    :
        Data with broadcast uncertainties.
    """
    if (
        not _has_variances(data)
        or uncertainty_broadcast_mode is None
        or uncertainty_broadcast_mode == UncertaintyBroadcastMode.fail
    ):
        return data

    if uncertainty_broadcast_mode == UncertaintyBroadcastMode.drop:
        return _without_variances(data)
    raise NotImplementedError(
        'Broadcasting uncertainties with the upper bound mode is not implemented'
    )


def _has_variances(data: sc.DataArray) -> bool:
    return (
        data.bins is not None and data.bins.constituents['data'].variances is not None
    ) or data.variances is not None


def _without_variances(data: sc.DataArray) -> sc.DataArray:
    out = data.copy(deep=False)
    if out.bins is not None:
        out.bins.constituents['data'].variances = None
    else:
        out.variances = None
    return out


providers = (drop_variances,)
"""Sciline providers for handling statistical uncertainties."""
