# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Tools for handling statistical uncertainties."""

from typing import TypeVar, Union

import scipp as sc

from .logging import get_logger

T = TypeVar("T", bound=Union[sc.Variable, sc.DataArray])


def drop_variances(da: T, *, reference: T, name: str = '', threshold: float = 0.1) -> T:
    # TODO cite our paper
    """Return the input values without variances.
    This assumes that ``da`` will be used as a normalization
    factor for ``reference``. That is, downstream code will perform an operation
    along the lines of ``reference / da``. It furthermore assumes that ``da``
    needs to be broadcast to the shape of ``reference`` in that operation.
    This broadcast would be forbidden with variances in Scipp, hence the need
    to drop them.
    A message will be logged showing :math:`\\alpha`, which indicates whether dropping
    variances might lead to underestimated uncertainties in the final result.

    Parameters
    ----------
    da:
        Data to remove variances from.
    reference:
        Assume that ``da`` will be used to normalize ``reference``.
    name:
        Optional name for ``da``. Shown in log messages.
    threshold:
        Raise an error if :math:`\\alpha` is above this value,
        log the value of :math:`\\alpha` on 'info' level otherwise.

    Returns
    -------
    :
        ``sc.values(da)``
    """
    alpha = alpha_ratio(numerator=reference, denominator=da)
    logger = get_logger()
    if alpha < threshold:
        name_str = f" of '{name}'" if name else ''
        logger.info(f'Dropping variances{name_str}. '
                    f'Total counts divided by reference: alpha = {alpha}.')
    else:
        raise ValueError(
            f'Cannot drop variances of data: alpha = {alpha} > {threshold}.')
    return sc.values(da)


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
