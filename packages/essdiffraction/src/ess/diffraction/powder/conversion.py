# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Coordinate transformations for powder diffraction.
"""

import uuid
from typing import Optional

import scipp as sc

from ..logging import get_logger
from ..types import CalibrationData, DspacingData, NormalizedByProtonCharge, RunType
from .correction import merge_calibration


def _dspacing_from_diff_calibration_generic_impl(t, t0, a, c):
    """
    This function implements the solution to
      t = a * d^2 + c * d + t0
    for a != 0.
    It uses the following way of expressing the solution with an order of operations
    that is optimized for low memory usage.
      d = (sqrt([x-t0+t] / x) - 1) * c / (2a)
      x = c^2 / (4a)
    """
    x = c**2 / (4 * a)
    out = (x - t0) + t
    out /= x
    del x
    sc.sqrt(out, out=out)
    out -= 1
    out *= c / (2 * a)
    return out


def _dspacing_from_diff_calibration_a0_impl(t, t0, c):
    """
    This function implements the solution to
      t = a * d^2 + c * d + t0
    for a == 0.
    """
    out = t - t0
    out /= c
    return out


def dspacing_from_diff_calibration(
    tof: sc.Variable,
    tzero: sc.Variable,
    difa: sc.Variable,
    difc: sc.Variable,
    _tag_positions_consumed: sc.Variable,
) -> sc.Variable:
    r"""
    Compute d-spacing from calibration parameters.

    d-spacing is the positive solution of

    .. math:: \mathsf{tof} = \mathsf{DIFA} * d^2 + \mathsf{DIFC} * d + t_0

    This function can be used with :func:`scipp.transform_coords`.

    See Also
    --------
    ess.diffraction.conversions.to_dspacing_with_calibration
    """
    if sc.all(difa == sc.scalar(0.0, unit=difa.unit)).value:
        return _dspacing_from_diff_calibration_a0_impl(tof, tzero, difc)
    return _dspacing_from_diff_calibration_generic_impl(tof, tzero, difa, difc)


def _restore_tof_if_in_wavelength(data: sc.DataArray) -> sc.DataArray:
    if 'wavelength' not in data.dims:
        return data

    get_logger().info("Discarding coordinate 'wavelength' in favor of 'tof'.")
    temp_name = uuid.uuid4().hex
    aux = data.transform_coords(
        temp_name,
        {temp_name: lambda wavelength, tof: tof},
        keep_inputs=False,
        quiet=True,
    )
    return aux.transform_coords(
        'tof', {'tof': temp_name}, keep_inputs=False, quiet=True
    )


def _consume_positions(position, sample_position, source_position):
    _ = position
    _ = sample_position
    _ = source_position
    return sc.scalar(0)


def to_dspacing_with_calibration(
    data: NormalizedByProtonCharge[RunType],
    calibration: Optional[CalibrationData] = None,
) -> DspacingData[RunType]:
    """
    Transform coordinates to d-spacing from calibration parameters.

    Computes d-spacing from time-of-flight stored in `data`.

    Attention
    ---------
    `data` may have a wavelength coordinate and dimension,
    but those are discarded.
    Only the stored time-of-flight is used, that is, any modifications to
    the wavelength coordinate after it was computed from time-of-flight are lost.

    Raises
    ------
    KeyError
        If `data` does not contain a 'tof' coordinate.

    Parameters
    ----------
    data:
        Input data in tof or wavelength dimension.
        Must have a tof coordinate.
    calibration:
        Calibration data. If given, use it for the conversion.
        Otherwise, the calibration data must be stored in `data`.

    Returns
    -------
    :
        A DataArray with the same data as the input and a 'dspacing' coordinate.

    See Also
    --------
    ess.diffraction.conversions.dspacing_from_diff_calibration
    """
    if calibration is not None:
        out = merge_calibration(into=data, calibration=calibration)
    else:
        out = data

    out = _restore_tof_if_in_wavelength(out)
    graph = {
        'dspacing': dspacing_from_diff_calibration,
    }

    if 'position' in out.coords:
        graph['_tag_positions_consumed'] = _consume_positions
    else:
        out.coords['_tag_positions_consumed'] = sc.scalar(0)

    out = out.transform_coords('dspacing', graph=graph, keep_intermediate=False)
    out.coords.pop('_tag_positions_consumed', None)
    return DspacingData[RunType](out)


providers = (to_dspacing_with_calibration,)
"""Sciline providers for coordinate transformations."""
