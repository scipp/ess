# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Tools for handling statistical uncertainties."""

from .types import RawData, RawDataWithVariances, VanadiumRun


def drop_variances(data: RawDataWithVariances[VanadiumRun]) -> RawData[VanadiumRun]:
    res = data.copy(deep=False)
    if res.bins is not None:
        res.bins.constituents['data'].variances = None
    else:
        res.variances = None
    return RawData[VanadiumRun](res)


providers = (drop_variances,)
"""Sciline providers for handling statistical uncertainties."""
