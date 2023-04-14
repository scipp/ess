# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Tools for handling statistical uncertainties."""

from typing import TypeVar, Union

import scipp as sc

T = TypeVar("T", bound=Union[sc.Variable, sc.DataArray])


def alpha_ratio(numerator: T, denominator: T) -> float:
    """
    .. math::
       \\alpha = \\frac{\\sum_{i} \\text{var}_{i}(b) a_{i}^{2}}{\\text{var}_{i}(a) b_{i}^{2}}

    where :math:`a` is the numerator and :math:`b` the denominator.

    Parameters
    ----------
    numerator:
        Numerator of the ratio.
    denominator:
        Denominator of the ratio.
    """  # noqa: E501
    alpha = sc.sum(sc.variances(denominator) * sc.values(numerator)**2).data / sc.sum(
        sc.variances(numerator) * sc.values(denominator)**2).data
    if alpha.unit != 'one':
        raise sc.UnitError(
            'Cannot compare counts, the reference has a different unit from the data.')
    return float(alpha.value)
