# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Tools for handling statistical uncertainties."""

import scipp as sc
from typing import Union, TypeVar
from .logging import get_logger


T = TypeVar("T", bound=Union[sc.Variable, sc.DataArray])

def drop_variances(da: T, *, reference: Union[sc.Variable, sc.DataArray], name: str='', threshold: float=1.0) -> T:
    # TODO cite our paper
    """Return the input values without variances.

    This assumes that ``da`` will be used as a normalization
    factor for ``reference``. That is, downstream code will perform an operation
    along the lines of ``reference / da``. It furthermore assumes that ``da``
    needs to be broadcast to the shape of ``reference`` in that operation.
    This broadcast would be forbidden with variances in Scipp, hence the need
    to drop them.

    A message will be logged showing :math:`\\alpha`, the ratio
    ``reference.sum() / da.sum()``, which indicates whether dropping variances
    might lead to underestimated uncertainties in the final result.

    Parameters
    ----------
    da:
        Data to remove variances from.
    reference:
        Assume that ``da`` will be used to normalize ``reference``.
    name:
        Optional name for ``da``. Shown in log messages.
    threshold:
        Log a warning if :math:`\\alpha` is above this value,
        log on 'info' level otherwise.

    Returns
    -------
    :
        ``sc.values(da)``
    """
    alpha = _compute_alpha(numerator=reference, denominator=da)
    logger = get_logger()
    msg = 'Dropping variances%s. Total counts divided by reference: alpha = %f'
    if alpha < threshold:
        logger.info(msg, f" of '{name}'" if name else '', alpha)
    else:
        logger.warning(msg + ' This value may be too large and the uncertainties of the result may be underestimated.',
                       f"of '{name}'" if name else '', alpha)
    return sc.values(da)


def _compute_alpha(numerator: Union[sc.Variable, sc.DataArray], denominator: Union[sc.Variable, sc.DataArray]) -> float:
    alpha = numerator.sum().data / denominator.sum().data
    if alpha.unit != 'one':
        raise sc.UnitError('Cannot compare counts, the reference has a different unit from the data.')
    return float(alpha.value)
