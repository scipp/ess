# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Histogramming helpers for POWTEX texture reduction."""

import math

import numpy as np
import scipp as sc
from typing import cast

from ..powtex.types import RunType, SampleRun
from .conversion import get_cartesian_from_cylindrical
from .types import (
    BinningSettings,
    DetectorViewRawData,
    DspacingBinWidth,
    DspacingCylSurfaceDetector,
    FlatViewRawData,
    GammaBinningData,
    GammaBinWidth,
    GammaSlicesViewData,
    GammaSlicesViewRawData,
    NormalisedTexture3D,
    NormalisedTexture2D,
    SampleRawData2D,
    Texture3D,
    TwoThetaBinningData,
    TwoThetaBinWidth,
    TwoThetaRingsViewData,
    TwoThetaRingsViewRawData,
)


def generate_gamma_binning_data(
    bin_width: GammaBinWidth,
    binning_settings: BinningSettings,
) -> GammaBinningData:
    """Generate gamma binning data with the POWTEX detector gap removed."""
    excluded_start = binning_settings.gamma_cut_start
    excluded_end = binning_settings.gamma_cut_end
    drop_partial_bins = binning_settings.drop_partial_bins

    gap_center = (excluded_start + excluded_end) / 2.0

    width = float(bin_width.to(unit="deg").value)
    gap_start = float(excluded_start.to(unit="deg").value)
    gap_end = float(excluded_end.to(unit="deg").value)
    gap_mid = float(gap_center.to(unit="deg").value)

    if width <= 0.0:
        raise ValueError("bin_width must be positive.")

    if not 0.0 <= gap_start < gap_end <= 360.0:
        raise ValueError("Expected 0 deg <= excluded_start < excluded_end <= 360 deg.")

    # region_1 corresponds to the interval [0 deg, excluded_start)
    # region_2 corresponds to the interval [excluded_end, 360 deg)
    num_bins_region_1 = math.floor(gap_start / width)
    num_bins_region_2 = math.floor((360.0 - gap_end) / width)

    region_1_end = num_bins_region_1 * width
    region_2_start = 360.0 - num_bins_region_2 * width

    dim = "gamma"
    region_1_edges = sc.linspace(
        dim, 0.0, region_1_end, num=num_bins_region_1 + 1, unit="deg"
    )
    region_2_edges = sc.linspace(
        dim, region_2_start, 360.0, num=num_bins_region_2 + 1, unit="deg"
    )

    if not drop_partial_bins:
        if not math.isclose(region_1_end, gap_start, rel_tol=1.0e-12, abs_tol=1.0e-12):
            region_1_edges = sc.concat(
                [region_1_edges, sc.array(dims=[dim], values=[gap_start], unit="deg")],
                dim=dim,
            )
        if not math.isclose(region_2_start, gap_end, rel_tol=1.0e-12, abs_tol=1.0e-12):
            region_2_edges = sc.concat(
                [sc.array(dims=[dim], values=[gap_end], unit="deg"), region_2_edges],
                dim=dim,
            )

    return GammaBinningData(
        sc.DataGroup({"region_1": region_1_edges, "region_2": region_2_edges})
    )


def generate_two_theta_binning_data(
    bin_width: TwoThetaBinWidth,
    binning_settings: BinningSettings,
) -> TwoThetaBinningData:
    """Generate two-theta bin edges."""
    angle_unit = bin_width.unit
    drop_partial_bins = binning_settings.drop_partial_bins
    min_angle = binning_settings.two_theta_min.to(unit=angle_unit)
    max_angle = binning_settings.two_theta_max.to(unit=angle_unit)
    mid_angle = (min_angle + max_angle) * 0.5

    if not (float(min_angle.value) < float(mid_angle.value) < float(max_angle.value)):
        raise ValueError("Expected min_angle < mid_angle < max_angle.")

    dim = "two_theta"
    lower_edges = generate_bin_edges_reverse(
        max_value=mid_angle,
        min_value=min_angle,
        bin_width=bin_width,
        drop_partial_bins=drop_partial_bins,
    )
    upper_edges = generate_bin_edges(
        min_value=mid_angle,
        max_value=max_angle,
        bin_width=bin_width,
        drop_partial_bins=drop_partial_bins,
    )
    # bin_centers = (
    #     bin_edges[dim, :-1] + bin_edges[dim, 1:]
    # ) / 2.0

    return TwoThetaBinningData(sc.concat([lower_edges, upper_edges], dim=dim))


def generate_bin_edges(
    min_value: sc.Variable,
    max_value: sc.Variable,
    bin_width: sc.Variable,
    drop_partial_bins: bool = True,
) -> sc.Variable:
    """Generate ascending bin edges starting at ``min_value``."""
    angle_unit = bin_width.unit
    min_value = min_value.to(unit=angle_unit)
    max_value = max_value.to(unit=angle_unit)

    lower = float(min_value.value)
    upper = float(max_value.value)
    width = float(bin_width.value)

    if width <= 0.0:
        raise ValueError("bin_width must be positive.")
    if lower >= upper:
        raise ValueError("min_value must be smaller than max_value.")

    n_complete = math.floor((upper - lower) / width)
    complete_end = lower + n_complete * width

    dim = "two_theta"
    edges = sc.linspace(dim, lower, complete_end, num=n_complete + 1, unit=angle_unit)
    if not drop_partial_bins and not math.isclose(complete_end, upper):
        edges = sc.concat(
            [edges, sc.array(dims=[dim], values=[upper], unit=angle_unit)], dim=dim
        )
    return edges


def generate_bin_edges_reverse(
    max_value: sc.Variable,
    min_value: sc.Variable,
    bin_width: sc.Variable,
    drop_partial_bins: bool = True,
) -> sc.Variable:
    """Generate ascending bin edges anchored at ``max_value``."""
    unit = bin_width.unit
    min_value = min_value.to(unit=unit)
    max_value = max_value.to(unit=unit)

    lower = float(min_value.value)
    upper = float(max_value.value)
    width = float(bin_width.value)

    if width <= 0.0:
        raise ValueError("bin_width must be positive.")
    if lower >= upper:
        raise ValueError("min_value must be smaller than max_value.")

    n_complete = math.floor((upper - lower) / width)
    dim = "two_theta"
    if n_complete == 0:
        if drop_partial_bins:
            return sc.array(dims=[dim], values=[], unit=unit)
        return sc.array(dims=[dim], values=[lower], unit=unit)

    complete_start = upper - n_complete * width

    # Generate directly in ascending order.
    edges = sc.linspace(dim, complete_start, upper - width, num=n_complete, unit=unit)
    if not drop_partial_bins and not math.isclose(complete_start, lower):
        edges = sc.concat(
            [sc.array(dims=[dim], values=[lower], unit=unit), edges], dim=dim
        )
    return edges


def hist_3d(
    data: DspacingCylSurfaceDetector[RunType],
    gamma_binning: GammaBinningData,
    two_theta_binning: TwoThetaBinningData,
    dspacing_bin_width: DspacingBinWidth,
) -> Texture3D[RunType]:
    """Histogram projected detector data to texture coordinates."""

    if data is None:
        return cast(Texture3D[RunType], None)

    dmin = sc.min(data.coords["dspacing"])
    dmax = sc.max(data.coords["dspacing"])
    dspacing_binning = sc.arange(
        "dspacing",
        dmin,
        dmax + dspacing_bin_width,
        dspacing_bin_width,
        unit="angstrom",
    )
    histogrammed_data_1 = data.hist(
        gamma=gamma_binning["region_1"],
        two_theta=two_theta_binning,
        dspacing=dspacing_binning,
    )
    histogrammed_data_2 = data.hist(
        gamma=gamma_binning["region_2"],
        two_theta=two_theta_binning,
        dspacing=dspacing_binning,
    )
    return Texture3D(
        sc.DataGroup({"region_1": histogrammed_data_1, "region_2": histogrammed_data_2})
    )


def hist_2d(data: Texture3D[SampleRun]) -> SampleRawData2D:
    """Histogram projected detector data to texture coordinates."""
    return SampleRawData2D(
        sc.DataGroup(
            {
                "region_1": data["region_1"].sum(dim="dspacing"),
                "region_2": data["region_2"].sum(dim="dspacing")
            }
        )
    )


def extract_detector_view_data(data: SampleRawData2D) -> DetectorViewRawData:
    """Return a scipp DataArray with data for the detector view."""
    histogrammed_2d_1 = data["region_1"]
    histogrammed_2d_2 = data["region_2"]

    gamma_bin_edges_1 = histogrammed_2d_1.coords["gamma"]
    gamma_bin_edges_2 = histogrammed_2d_2.coords["gamma"]

    # Concatenate gamma bin edges without duplicating the shared (360 deg, 0 deg) boundary.
    gamma_bin_edges = sc.concat(
        [gamma_bin_edges_2["gamma", :-1], gamma_bin_edges_1],
        dim="gamma",
    )

    # Both regions must have identical two-theta bin edges.
    two_theta_bin_edges = histogrammed_2d_1.coords["two_theta"]

    counts_1 = histogrammed_2d_1.data
    counts_2 = histogrammed_2d_2.data

    counts = sc.concat([counts_2, counts_1], dim="gamma")

    projection_radius = histogrammed_2d_1.coords["projection_radius"]

    x_bin_edges, y_bin_edges, z_bin_edges = get_cartesian_from_cylindrical(
        projection_radius, gamma_bin_edges, two_theta_bin_edges
    )

    return DetectorViewRawData(
        sc.DataArray(
            counts,
            coords={
                "gamma": gamma_bin_edges,
                "two_theta": two_theta_bin_edges,
                "x": x_bin_edges,
                "y": y_bin_edges,
                "z": z_bin_edges,
                "projection_radius": projection_radius,
            },
        )
    )


def extract_flat_view_data(data: SampleRawData2D) -> FlatViewRawData:
    """Return a scipp DataArray with data for a raw flattened detector view.
    Note, in order to properly visualize the gap in the detector, the gamma
    bin edges are modified to include a gap in the middle and one needs to
    work with bin edge coordinates for proper visualization."""

    gamma_region_1_last_edge = data["region_1"].coords["gamma"][-1]
    gamma_region_2_first_edge = data["region_2"].coords["gamma"][0]
    gamma_gap = (gamma_region_1_last_edge + gamma_region_2_first_edge) / 2.0

    gamma_bin_edges = sc.concat(
        [data["region_1"].coords["gamma"],  gamma_gap, data["region_2"].coords["gamma"]],
        dim="gamma",
    )

    two_theta_bin_edges = data["region_1"].coords["two_theta"]

    counts_1 = data["region_1"].data
    counts_2 = data["region_2"].data
    # Create a placeholder for the gap in the middle with NaN values.
    # The gap in the middle generates two bins with NaN values:
    # from the last bin edge of region_1 to the gap point and from
    # the gap point to the first bin edge of region_2.
    counts_gap = sc.full(dims=["gamma", "two_theta"], shape=[2, counts_1.shape[1]], value=np.nan, unit=counts_1.unit)

    counts = sc.concat([counts_1, counts_gap, counts_2], dim="gamma")

    return FlatViewRawData(
        sc.DataArray(
            counts,
            coords={
                "gamma": gamma_bin_edges,
                "two_theta": two_theta_bin_edges
            },
        )
    )


def _two_theta_rings_data(data: sc.DataGroup) -> sc.DataArray:
    """Return a scipp DataArray with data for a raw two-theta rings view"""
    histogrammed_2d_1 = data["region_1"]
    histogrammed_2d_2 = data["region_2"]

    gamma_bin_centers_1 = sc.midpoints(histogrammed_2d_1.coords["gamma"])
    gamma_bin_centers_2 = sc.midpoints(histogrammed_2d_2.coords["gamma"])
    # TODO: check that dims match
    gamma_bin_centers = sc.concat(
        [gamma_bin_centers_1, gamma_bin_centers_2], dim="gamma"
    )

    two_theta_bin_centers = sc.midpoints(histogrammed_2d_1.coords["two_theta"])

    rings_radius = sc.full_like(gamma_bin_centers, 800)

    counts_1 = histogrammed_2d_1.data
    counts_2 = histogrammed_2d_2.data
    counts = sc.concat([counts_1, counts_2], dim="gamma")

    gamma_widths_1 = sc.abs(
        data["region_1"].coords["gamma"]["gamma", 1:] - data["region_1"].coords["gamma"]["gamma", :-1]
    )
    gamma_widths_2 = sc.abs(
        data["region_2"].coords["gamma"]["gamma", 1:] - data["region_2"].coords["gamma"]["gamma", :-1]
    )
    gamma_widths = sc.concat(
        [gamma_widths_1, gamma_widths_2],
        dim="gamma"
    )

    return sc.DataArray(
        counts,
        coords={
            "gamma": gamma_bin_centers,
            "gamma_width": gamma_widths,
            "two_theta": two_theta_bin_centers,
            "r": rings_radius
        },
    )


def extract_two_theta_rings_view_raw_data(
    data: SampleRawData2D
) -> TwoThetaRingsViewRawData:
    return TwoThetaRingsViewRawData(_two_theta_rings_data(data))


def extract_two_theta_rings_view_data(
    data: NormalisedTexture2D,
) -> TwoThetaRingsViewData:
    return TwoThetaRingsViewData(_two_theta_rings_data(data))


def _gamma_slices_view_data(data: sc.DataGroup) -> sc.DataArray:
    """Return a scipp DataArray with data for a raw gamma-slices view."""
    gamma_region_1_last_edge = data["region_1"].coords["gamma"][-1]
    gamma_region_2_first_edge = data["region_2"].coords["gamma"][0]
    gamma_gap = (gamma_region_1_last_edge + gamma_region_2_first_edge) / 2.0

    gamma_bin_edges = sc.concat(
        [data["region_1"].coords["gamma"],  gamma_gap, data["region_2"].coords["gamma"]],
        dim="gamma",
    )

    two_theta_bin_edges = data["region_1"].coords["two_theta"]

    gamma_bin_centers = sc.midpoints(gamma_bin_edges)
    two_theta_bin_centers = sc.midpoints(two_theta_bin_edges)

    counts_1 = data["region_1"].data
    counts_2 = data["region_2"].data
    # Create a placeholder for the gap in the middle with NaN values.
    # The gap in the middle generates two bins with NaN values:
    # from the last bin edge of region_1 to the gap point and from
    # the gap point to the first bin edge of region_2.
    counts_gap = sc.full(dims=["gamma", "two_theta"], shape=[2, counts_1.shape[1]], value=np.nan, unit=counts_1.unit)

    counts = sc.concat([counts_1, counts_gap, counts_2], dim="gamma")

    return sc.DataArray(
        counts,
        coords={
            "gamma": gamma_bin_centers,
            "two_theta": two_theta_bin_centers
        },
    )


def extract_gamma_slices_view_raw_data(data: SampleRawData2D) -> GammaSlicesViewRawData:
    return GammaSlicesViewRawData(_gamma_slices_view_data(data))


def extract_gamma_slices_view_data(data: NormalisedTexture2D) -> GammaSlicesViewData:
    return GammaSlicesViewData(_gamma_slices_view_data(data))


def integrate_dspacing(data: NormalisedTexture3D) -> NormalisedTexture2D:
    """Histogram projected detector data to texture coordinates."""
    return NormalisedTexture2D(
        sc.DataGroup(
            {
                "region_1": data["region_1"].sum(dim="dspacing"),
                "region_2": data["region_2"].sum(dim="dspacing")
            }
        )
    )


providers = (
    generate_gamma_binning_data,
    generate_two_theta_binning_data,
    hist_3d,
    hist_2d,
    extract_detector_view_data,
    extract_flat_view_data,
    extract_two_theta_rings_view_data,
    extract_two_theta_rings_view_raw_data,
    extract_gamma_slices_view_data,
    extract_gamma_slices_view_raw_data,
    integrate_dspacing,
)
