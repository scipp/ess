# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Coordinate transformations for powder diffraction.
"""

import scipp as sc
import scippneutron as scn

from .calibration import OutputCalibrationData
from .correction import merge_calibration
from .logging import get_logger
from .types import (
    CalibrationData,
    DataWithScatteringCoordinates,
    DspacingData,
    ElasticCoordTransformGraph,
    FilteredData,
    IofDspacing,
    IofTof,
    MaskedData,
    MonitorTofData,
    MonitorType,
    RunType,
    WavelengthMonitor,
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
    data: sc.DataArray,
    calibration: sc.Dataset,
) -> sc.DataArray:
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
        Calibration data.

    Returns
    -------
    :
        A DataArray with the same data as the input and a 'dspacing' coordinate.

    See Also
    --------
    ess.powder.conversions.dspacing_from_diff_calibration
    """
    out = merge_calibration(into=data, calibration=calibration)
    out = _restore_tof_if_in_wavelength(out)

    graph = {"dspacing": _dspacing_from_diff_calibration}
    # `_dspacing_from_diff_calibration` does not need positions but conceptually,
    # the conversion maps from positions to d-spacing.
    # The mechanism with `_tag_positions_consumed` is meant to ensure that,
    # if positions are present, they are consumed (mad unaligned or dropped)
    # by the coordinate transform similarly to `to_dspacing_with_positions`.
    if "position" in out.coords or (
        out.bins is not None and "position" in out.bins.coords
    ):
        graph["_tag_positions_consumed"] = _consume_positions
    else:
        graph["_tag_positions_consumed"] = lambda: sc.scalar(0)
    out = out.transform_coords("dspacing", graph=graph, keep_intermediate=False)
    out.coords.pop("_tag_positions_consumed", None)
    return DspacingData[RunType](out)


def powder_coordinate_transformation_graph() -> ElasticCoordTransformGraph:
    """
    Generate a coordinate transformation graph for powder diffraction.

    Returns
    -------
    :
        A dictionary with the graph for the transformation.
    """
    return ElasticCoordTransformGraph(
        {
            **scn.conversion.graph.beamline.beamline(scatter=True),
            **scn.conversion.graph.tof.elastic("tof"),
        }
    )


def _restore_tof_if_in_wavelength(data: sc.DataArray) -> sc.DataArray:
    out = data.copy(deep=False)
    outer = out.coords.pop("wavelength", None)
    if out.bins is not None:
        binned = out.bins.coords.pop("wavelength", None)
    else:
        binned = None

    if outer is not None or binned is not None:
        get_logger().info("Discarded coordinate 'wavelength' in favor of 'tof'.")

    if "wavelength" in out.dims:
        out = out.rename_dims(wavelength="tof")
    return out


def add_scattering_coordinates_from_positions(
    data: FilteredData[RunType], graph: ElasticCoordTransformGraph
) -> DataWithScatteringCoordinates[RunType]:
    """
    Add ``wavelength`` and ``two_theta`` coordinates to the data.
    The input ``data`` must have a ``tof`` coordinate, as well as the necessary
    positions of the beamline components (source, sample, detectors) to compute
    the scattering coordinates.

    Parameters
    ----------
    data:
        Input data with a ``tof`` coordinate.
    graph:
        Coordinate transformation graph.
    """
    out = data.transform_coords(
        ["two_theta", "wavelength", "Ltotal"],
        graph=graph,
        keep_intermediate=False,
    )
    return DataWithScatteringCoordinates[RunType](out)


def convert_to_dspacing(
    data: MaskedData[RunType],
    graph: ElasticCoordTransformGraph,
    calibration: CalibrationData,
) -> DspacingData[RunType]:
    if calibration is None:
        out = data.transform_coords(["dspacing"], graph=graph, keep_intermediate=False)
    else:
        out = to_dspacing_with_calibration(data, calibration=calibration)
        for key in ("wavelength", "two_theta"):
            if key in out.coords.keys():
                out.coords.set_aligned(key, False)
    out.bins.coords.pop("tof", None)
    out.bins.coords.pop("wavelength", None)
    return DspacingData[RunType](out)


def convert_reduced_to_tof(
    data: IofDspacing, calibration: OutputCalibrationData
) -> IofTof:
    return IofTof(
        data.transform_coords(tof=calibration.d_to_tof_transformer(), keep_inputs=False)
    )


def convert_monitor_to_wavelength(
    monitor: MonitorTofData[RunType, MonitorType],
) -> WavelengthMonitor[RunType, MonitorType]:
    graph = {
        **scn.conversion.graph.beamline.beamline(scatter=False),
        **scn.conversion.graph.tof.elastic("tof"),
    }
    return WavelengthMonitor[RunType, MonitorType](
        monitor.transform_coords("wavelength", graph=graph, keep_intermediate=False)
    )


providers = (
    powder_coordinate_transformation_graph,
    add_scattering_coordinates_from_positions,
    convert_to_dspacing,
    convert_reduced_to_tof,
    convert_monitor_to_wavelength,
)
