# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Coordinate transformations for powder diffraction.
"""

from typing import Optional

import scipp as sc
import scippneutron as scn

from .correction import merge_calibration
from .logging import get_logger
from .types import (
    CalibrationData,
    DspacingData,
    NormalizedByProtonCharge,
    RawSample,
    RawSource,
    RunType,
)


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


def _dspacing_from_diff_calibration(
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
    ess.powder.conversions.to_dspacing_with_calibration
    """
    if sc.all(difa == sc.scalar(0.0, unit=difa.unit)).value:
        return _dspacing_from_diff_calibration_a0_impl(tof, tzero, difc)
    return _dspacing_from_diff_calibration_generic_impl(tof, tzero, difa, difc)


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
    ess.powder.conversions.dspacing_from_diff_calibration
    """
    if calibration is not None:
        out = merge_calibration(into=data, calibration=calibration)
    else:
        out = data
    out = _restore_tof_if_in_wavelength(out)

    graph = {
        'dspacing': _dspacing_from_diff_calibration,
    }
    # `_dspacing_from_diff_calibration` does not need positions but conceptually,
    # the conversion maps from positions to d-spacing.
    # The mechanism with `_tag_positions_consumed` is meant to ensure that,
    # if positions are present, they are consumed (mad unaligned or dropped)
    # by the coordinate transform similarly to `to_dspacing_with_positions`.
    if 'position' in out.coords or (
        out.bins is not None and 'position' in out.bins.coords
    ):
        graph['_tag_positions_consumed'] = _consume_positions
    else:
        graph['_tag_positions_consumed'] = lambda: sc.scalar(0)
    out = out.transform_coords('dspacing', graph=graph, keep_intermediate=False)
    out.coords.pop('_tag_positions_consumed', None)
    return DspacingData[RunType](out)


def to_dspacing_with_positions(
    data: NormalizedByProtonCharge[RunType],
    *,
    sample: Optional[RawSample[RunType]] = None,
    source: Optional[RawSource] = None,
) -> DspacingData[RunType]:
    """
    Transform coordinates to d-spacing using detector positions.

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
    sample:
        Sample data with a position.
        If not given, ``data`` must contain a 'sample_position' coordinate.
    source:
        Source data with a position.
        If not given, ``data`` must contain a 'source_position' coordinate.

    Returns
    -------
    :
        A DataArray with the same data as the input and a 'dspacing' coordinate.
    """
    graph = {
        **scn.conversion.graph.beamline.beamline(scatter=True),
        **scn.conversion.graph.tof.elastic_dspacing('tof'),
    }
    if sample is not None:
        graph['sample_position'] = lambda: sample['position']
    if source is not None:
        graph['source_position'] = lambda: source['position']

    out = _restore_tof_if_in_wavelength(data)
    out = out.transform_coords('dspacing', graph=graph, keep_intermediate=False)
    # Add coords to ensure the result is the same whether sample or source are
    # coords in the input or separate function arguments.
    if sample is not None:
        out.coords['sample_position'] = sample['position']
        out.coords.set_aligned('sample_position', False)
    if source is not None:
        out.coords['source_position'] = source['position']
        out.coords.set_aligned('source_position', False)

    return DspacingData[RunType](out)


def _restore_tof_if_in_wavelength(data: sc.DataArray) -> sc.DataArray:
    out = data.copy(deep=False)
    outer = out.coords.pop('wavelength', None)
    if out.bins is not None:
        binned = out.bins.coords.pop('wavelength', None)
    else:
        binned = None

    if outer is not None or binned is not None:
        get_logger().info("Discarded coordinate 'wavelength' in favor of 'tof'.")

    if 'wavelength' in out.dims:
        out = out.rename_dims(wavelength='tof')
    return out


providers = (to_dspacing_with_calibration,)
"""Sciline providers for coordinate transformations."""
