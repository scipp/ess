# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..reflectometry import orso


def supermirror_calibration(
    data_array: sc.DataArray,
    m_value: sc.Variable = None,
    critical_edge: sc.Variable = None,
    alpha: sc.Variable = None,
) -> sc.Variable:
    """
    Calibrate supermirror measurements

    Parameters
    ----------
    data_array:
        Data array to get q-bins/values from.
    m_value:
        m-value for the supermirror. Defaults to 5.
    critical_edge:
        Supermirror critical edge. Defaults to 0.022 1/angstrom.
    alpha:
        Supermirror alpha value. Defaults to 0.25 / 0.088 angstrom.

    Returns
    -------
    :
        Calibrated supermirror measurement.
    """
    if m_value is None:
        m_value = sc.scalar(5, unit=sc.units.dimensionless)
    if critical_edge is None:
        critical_edge = 0.022 * sc.Unit('1/angstrom')
    if alpha is None:
        alpha = sc.scalar(0.25 / 0.088, unit=sc.units.angstrom)
    calibration = calibration_factor(data_array, m_value, critical_edge, alpha)
    data_array_cal = data_array * calibration
    try:
        data_array_cal.attrs['orso'].value.reduction.corrections += [
            'supermirror calibration'
        ]
    except KeyError:
        orso.not_found_warning()
    return data_array_cal


def calibration_factor(
    data_array: sc.DataArray,
    m_value: sc.Variable = None,
    critical_edge: sc.Variable = None,
    alpha: sc.Variable = None,
) -> sc.Variable:
    """
    Return the calibration factor for the supermirror.

    Parameters
    ----------
    data_array:
        Data array to get q-bins/values from.
    m_value:
        m-value for the supermirror. Defaults to 5.
    critical_edge:
        Supermirror critical edge. Defaults to 0.022 1/angstrom.
    alpha:
        Supermirror alpha value. Defaults to 0.25 / 0.088 angstrom.

    Returns
    -------
    :
        Calibration factor at the midpoint of each Q-bin.
    """
    if m_value is None:
        m_value = sc.scalar(5, unit=sc.units.dimensionless)
    if critical_edge is None:
        critical_edge = 0.022 * sc.Unit('1/angstrom')
    if alpha is None:
        alpha = sc.scalar(0.25 / 0.088, unit=sc.units.angstrom)
    q = data_array.coords['Q']
    if data_array.coords.is_edges('Q'):
        q = sc.midpoints(q)
    max_q = m_value * critical_edge
    lim = (q < critical_edge).astype(float)
    lim.unit = 'one'
    nq = 1.0 / (1.0 - alpha * (q - critical_edge))
    calibration_factor = sc.where(q < max_q, lim + (1 - lim) * nq, sc.scalar(1.0))
    return calibration_factor
