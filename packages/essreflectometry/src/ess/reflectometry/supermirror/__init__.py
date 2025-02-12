# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
# flake8: noqa: F401
import numpy as np
import scipp as sc

from .types import Alpha, CriticalEdge, MValue


def supermirror_reflectivity(
    q: sc.Variable, c: CriticalEdge, m: MValue, alpha: Alpha
) -> sc.Variable:
    """
    Returns the reflectivity of the supermirror.
    For ``q`` outside of the region of known reflectivity
    this function returns ``nan``.

    Parameters
    ----------
    q:
        Momentum transfer.
    m_value:
        m-value for the supermirror.
    critical_edge:
        Supermirror critical edge.
    alpha:
        Supermirror alpha value.

    Returns
    -------
    :
        Reflectivity of the supermirror at q.
    """
    return sc.where(
        q < c,
        sc.scalar(1.0),
        sc.where(q < m * c, sc.scalar(1) - alpha * (q - c), sc.scalar(np.nan)),
    )
