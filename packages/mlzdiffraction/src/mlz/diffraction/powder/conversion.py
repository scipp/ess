# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Coordinate conversions for POWTEX powder reduction."""

from typing import cast

import numpy as np
import scipp as sc
import scippneutron as scn

from ..powtex.types import (
    CalibrationData,
    RunType,
    TofDetector,
    WavelengthDetector,
)
from .types import ElasticCoordTransformGraph


# Copied from ess.powder.correction, which might be modified
# in future, so we keep a copy here for now to avoid breaking
# the MLZ workflow
def merge_calibration(*, into: sc.DataArray, calibration: sc.Dataset) -> sc.DataArray:
    """Return a copy of data with calibration parameters attached."""
    for name, coord in calibration.coords.items():
        if not sc.identical(into.coords[name], coord):
            raise ValueError(
                f"Coordinate {name} of calibration and target dataset do not agree."
            )
    out = into.copy(deep=False)
    for name in ("difa", "difc", "tzero"):
        if name in out.coords:
            raise ValueError(
                f"Cannot add calibration parameter '{name}' to data, "
                "there already is metadata with the same name."
            )
        out.coords[name] = calibration[name].data
    if "calibration" in out.masks:
        raise ValueError(
            "Cannot add calibration mask 'calibration' to data, "
            "there already is a mask with the same name."
        )
    if (mask := calibration.get("mask")) is not None:
        out.masks["calibration"] = mask.data
    return out


# Copied from ess.powder.conversion, which might be modified
# in future, so we keep a copy here for now to avoid breaking
# the MLZ workflow
def _dspacing_from_diff_calibration_generic_impl(
    tof: sc.Variable, tzero: sc.Variable, difa: sc.Variable, difc: sc.Variable
) -> sc.Variable:
    """Compute d-spacing from POWGEN-style calibration coefficients."""
    x = difc**2 / (4.0 * difa)
    out = (x - tzero) + tof
    out /= x
    del x
    sc.sqrt(out, out=out)
    out -= 1.0
    out *= difc / (2.0 * difa)
    return out


# Copied from ess.powder.conversion, which might be modified
# in future, so we keep a copy here for now to avoid breaking
# the MLZ workflow
def _dspacing_from_diff_calibration_a0_impl(
    tof: sc.Variable, tzero: sc.Variable, difc: sc.Variable
) -> sc.Variable:
    out = tof - tzero
    out /= difc
    return out


# Copied from ess.powder.conversion, which might be modified
# in future, so we keep a copy here for now to avoid breaking
# the MLZ workflow
def _dspacing_from_diff_calibration(
    tof: sc.Variable,
    tzero: sc.Variable,
    difa: sc.Variable,
    difc: sc.Variable,
) -> sc.Variable:
    if sc.all(difa == sc.scalar(0.0, unit=difa.unit)).value:
        return _dspacing_from_diff_calibration_a0_impl(tof, tzero, difc)
    return _dspacing_from_diff_calibration_generic_impl(tof, tzero, difa, difc)


# Copied from ess.powder.conversion.
# TODO:revisit in future, if this is needed in the MLZ workflow as a part of
# to_dspacing_with_calibration_mlz() or not
def _consume_positions(position, sample_position, source_position):
    _ = position
    _ = sample_position
    _ = source_position
    return sc.scalar(0)


def d_perp_from_lambda_and_theta(
    wavelength: sc.Variable, two_theta: sc.Variable
) -> sc.Variable:
    """Compute the dperp coordinate used by the POWTEX P2D workflow."""
    lambda_k = sc.scalar(1.0, unit="angstrom")
    return sc.sqrt(
        wavelength**2 - 2.0 * lambda_k**2 * sc.log(sc.cos(0.5 * two_theta))
    )


def d_dperp_from_tof_and_calibration_coeffs() -> dict:
    """Return a graph producing wavelength, d-spacing, and dperp."""
    return {
        "wavelength": scn.conversion.tof.wavelength_from_tof,
        "dspacing": _dspacing_from_diff_calibration_generic_impl,
        "dperp": d_perp_from_lambda_and_theta,
    }


def to_dspacing_with_calibration(
    data: sc.DataArray, calibration: sc.Dataset
) -> sc.DataArray:
    """Transform detector time-of-flight data to calibrated d-spacing."""
    out = merge_calibration(into=data, calibration=calibration)

    graph = {"tof": scn.conversion.tof.tof_from_wavelength}
    out = out.transform_coords("tof", graph=graph, keep_intermediate=False)

    pos_graph = {"dspacing": _dspacing_from_diff_calibration}
    if "position" in out.coords or (
        out.bins is not None and "position" in out.bins.coords
    ):
        pos_graph["_tag_positions_consumed"] = _consume_positions
    else:
        pos_graph["_tag_positions_consumed"] = lambda: sc.scalar(0)
    out = out.transform_coords("dspacing", graph=pos_graph, keep_intermediate=False)
    out.coords.pop("_tag_positions_consumed", None)
    for key in (
        "difa",
        "difc",
        "tzero",
        "position",
        "sample_position",
        "source_position",
        "Ltotal",
    ):
        out.coords.pop(key, None)
    return out


def p2d_coordinate_transformation_graph() -> ElasticCoordTransformGraph:
    """Generate the coordinate transformation graph for P2D reduction."""
    return ElasticCoordTransformGraph(
        {
            **scn.conversion.graph.beamline.beamline(scatter=True),
            **d_dperp_from_tof_and_calibration_coeffs(),
        }
    )


def add_scattering_coordinates_from_positions(
    data: TofDetector[RunType],
    graph: ElasticCoordTransformGraph,
    calibration: CalibrationData,
) -> WavelengthDetector[RunType]:
    """Add wavelength, two-theta, d-spacing, and dperp coordinates."""
    if data is None:
        return cast(WavelengthDetector[RunType], None)
    data = data.transform_coords(
        ["two_theta", "wavelength", "Ltotal", "dperp"],
        graph=graph,
        keep_intermediate=False,
    )
    out = convert_to_dspacing(data, graph, calibration)
    return WavelengthDetector[RunType](out)


def convert_to_dspacing(
    data: sc.DataArray,
    graph: ElasticCoordTransformGraph,
    calibration: CalibrationData,
) -> sc.DataArray:
    """Convert detector data to d-spacing, using calibration when available."""
    if calibration is None:
        # TODO: generalize the graph to include dperp
        out = data.transform_coords(["dspacing"], graph=graph, keep_intermediate=False)
    else:
        out = to_dspacing_with_calibration(data, calibration=calibration)
        for key in ("wavelength", "two_theta", "dperp"):
            if key in out.coords:
                out.coords.set_aligned(key, False)
    return out


def sc_wright_omega_real(
    d: sc.Variable, dp: sc.Variable, lam_k: sc.Variable
) -> sc.DataArray:
    """Compute the Wright omega function used by the P2D transform."""
    from scipy.special import lambertw

    argument = (
        4.0 * d**2 - lam_k**2 * sc.log(0.25 * lam_k**2 / d**2) - dp**2
    )
    argument /= lam_k**2
    values = np.real(lambertw(np.exp(argument.values), 0))
    omega = sc.array(dims=argument.dims, values=values, unit=argument.unit)
    return sc.DataArray(data=omega, coords={d.dim: d, dp.dim: dp})


def lambda_from_d_dperp(
    d: sc.Variable, dp: sc.Variable, lam_k: sc.Variable
) -> sc.DataArray:
    """Compute wavelength from d-spacing and dperp."""
    return sc.sqrt(4.0 * d**2 - lam_k**2 * sc_wright_omega_real(d, dp, lam_k))


providers = (
    add_scattering_coordinates_from_positions,
    p2d_coordinate_transformation_graph,
)
