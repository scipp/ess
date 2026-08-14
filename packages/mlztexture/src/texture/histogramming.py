import math

import scipp as sc
import numpy as np

from ess.packages.mlztexture.src.texture.types import (
    GammaBinWidth,
    TwoThetaBinWidth,
    DspacingBinWidth,
    BinningSettings,
    DspacingCylSurfaceDetector,
    GammaBinningData,
    TwoThetaBinningData,
    DspacingBinningData,
    Texture3D,
    View1D,
    View2D,
    View3D,
)

from ess.packages.essdiffraction.src.ess.powder.types import (
    RunType,
)


def generate_gamma_binning_data(
    bin_width: GammaBinWidth,
    binning_settings: BinningSettings
) -> GammaBinningData:
    "ToDo"
    excluded_start = binning_settings.gamma_cut_start
    excluded_end = binning_settings.gamma_cut_end
    drop_partial_bins = binning_settings.drop_partial_bins
    include_visualization_gap = binning_settings.include_gamma_gap_bins

    gap_center = (excluded_start + excluded_end) / 2.0

    width = float(bin_width.to(unit="deg").value)
    gap_start = float(excluded_start.to(unit="deg").value)
    gap_end = float(excluded_end.to(unit="deg").value)
    gap_mid = float(gap_center.to(unit="deg").value)

    if width <= 0.0:
        raise ValueError("bin_width must be positive.")

    if not 0.0 <= gap_start < gap_end <= 360.0:
        raise ValueError("Expected 0 deg <= excluded_start < excluded_end <= 360 deg.")

    if include_visualization_gap and not gap_start < gap_mid < gap_end:
        raise ValueError("gap_center must lie inside the excluded interval.")

    # region_1 corresponds to the interval [0 deg, excluded_start)
    # region_2 corresponds to the interval [excluded_end, 360 deg)
    num_bins_region_1 = math.floor(gap_start / width)
    num_bins_region_2 = math.floor((360.0 - gap_end) / width)

    region_1_end = num_bins_region_1 * width
    region_2_start = 360.0 - num_bins_region_2 * width

    # The two principal linspace calls.
    dim = "gamma"
    region_1_edges = sc.linspace(dim, 0.0, region_1_end, num=num_bins_region_1 + 1, unit="deg")
    region_2_edges = sc.linspace(dim, region_2_start, 360.0, num=num_bins_region_2 + 1, unit="deg")

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

    # This part below probably should be moved to a separate function
    # region_1_centers = sc.midpoints(region_1_edges) #(region_1_edges[dim, :-1] + region_1_edges[dim, 1:]) / 2.0
    # region_2_centers = sc.midpoints(region_2_edges) #(region_2_edges[dim, :-1] + region_2_edges[dim, 1:]) / 2.0

    # if include_visualization_gap:
    #     if (region_1_centers.sizes[dim] == 0 or region_2_centers.sizes[dim] == 0):
    #         raise ValueError("Visualization-gap mode requires at least one bin on each side of the excluded interval.")

    #     visualization_gap_centers = sc.array(
    #         dims=[dim],
    #         values=[float(region_1_centers[dim, -1].value) + width, gap_mid, float(region_2_centers[dim, 0].value) - width],
    #         unit="deg",
    #     )

    #     bin_centers = sc.concat(
    #         [region_1_centers, visualization_gap_centers, region_2_centers],
    #         dim=dim,
    #     )
    # else:
    #     # Preserve the original high-angle-first ordering.
    #     bin_centers = sc.concat(
    #         [region_2_centers, region_1_centers],
    #         dim=dim,
    #     )
    out = sc.DataGroup({"region_1": region_1_edges, "region_2": region_2_edges})
    return GammaBinningData(out)

def generate_two_theta_binning_data(
    bin_width: TwoThetaBinWidth,
    binning_settings: BinningSettings
) -> TwoThetaBinningData:
    """ToDo"""
    angle_unit = bin_width.unit
    drop_partial_bins = binning_settings.drop_partial_bins
    min_angle = binning_settings.two_theta_min.to(unit=angle_unit)
    max_angle = binning_settings.two_theta_max.to(unit=angle_unit)
    mid_angle = (min_angle + max_angle) * 0.5

    if not (float(min_angle.value) < float(mid_angle.value) < float(max_angle.value)):
        raise ValueError("Expected min_angle < mid_angle < max_angle.")

    dim = "two_theta"
    lower_edges = generate_bin_edges_reverse(max_value=mid_angle, min_value=min_angle, bin_width=bin_width, drop_partial_bins=drop_partial_bins)

    upper_edges = generate_bin_edges(min_value=mid_angle, max_value=max_angle, bin_width=bin_width, drop_partial_bins=drop_partial_bins)

    bin_edges = sc.concat([lower_edges, upper_edges], dim=dim)

    # bin_centers = (
    #     bin_edges[dim, :-1] + bin_edges[dim, 1:]
    # ) / 2.0

    return TwoThetaBinningData(bin_edges)


def generate_bin_edges(
    min_value: sc.Variable,
    max_value: sc.Variable,
    bin_width: sc.Variable,
    drop_partial_bins: bool = True,
) -> sc.Variable:
    """
    Generate ascending bin edges starting at ``min_value``.

    Complete bins are anchored at ``min_value``. If the interval does not
    contain an integer number of bins, the final partial bin is either
    dropped or retained depending on ``drop_partial_bins``.
    """
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

    if (not drop_partial_bins and not math.isclose(complete_end, upper)):
        edges = sc.concat([edges, sc.array(dims=[dim], values=[upper], unit=angle_unit)], dim=dim)

    return edges


def generate_bin_edges_reverse(
    max_value: sc.Variable,
    min_value: sc.Variable,
    bin_width: sc.Variable,
    drop_partial_bins: bool = True
) -> sc.Variable:
    """
    ToDo
    """
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

    if (not drop_partial_bins and not math.isclose(complete_start, lower)):
        edges = sc.concat([sc.array(dims=[dim], values=[lower], unit=unit), edges], dim=dim)

    return edges


def hist_3d(
    data: DspacingCylSurfaceDetector[RunType],
    gamma_binning: GammaBinningData,
    two_theta_binning: TwoThetaBinningData,
    dspacing_bin_width: DspacingBinWidth
) -> Texture3D[RunType]:

    # Prepare conventional dspacing binning
    dmin = sc.min(data.coords["dspacing"])
    dmax = sc.max(data.coords["dspacing"])
    dspacing_binning = sc.arange("dspacing", dmin, dmax + dspacing_bin_width, dspacing_bin_width, unit="angstrom")

    # Perform the histogramming
    histogrammed_data_1 = data.hist(gamma=gamma_binning["region_1"], two_theta=two_theta_binning, dspacing=dspacing_binning)
    histogrammed_data_2 = data.hist(gamma=gamma_binning["region_2"], two_theta=two_theta_binning, dspacing=dspacing_binning)

    out = sc.DataGroup({"region_1": histogrammed_data_1, "region_2": histogrammed_data_2})

    return Texture3D(out)


def view_3d(
    data: Texture3D[RunType],
) -> View3D[RunType]:
    """
    ToDo
    """
    histogrammed_2d_1 = data["region_1"].sum(dim="dspacing")
    histogrammed_2d_2 = data["region_2"].sum(dim="dspacing")

    out = sc.DataGroup({"region_1": histogrammed_2d_1, "region_2": histogrammed_2d_2})
    return View3D[RunType](out)


def view_2d(
    data: Texture3D[RunType],
    gamma_binning: GammaBinningData,
    two_theta_binning: TwoThetaBinningData,
) -> View2D[RunType]:
    """
    ToDo
    """
    out = data
    # Perform the histogramming operation.
    # histogrammed_data_1 = data.hist(gamma=gamma_binning["region_1"], two_theta=two_theta_binning)
    # histogrammed_data_2 = data.hist(gamma=gamma_binning["region_2"], two_theta=two_theta_binning)

    # out = sc.DataGroup({"region_1": histogrammed_data_1, "region_2": histogrammed_data_2})
    return View2D[RunType](out)

def view_1d(
    data: Texture3D[RunType],
    gamma_binning: GammaBinningData,
    two_theta_binning: TwoThetaBinningData,
) -> View1D[RunType]:
    """
    ToDo
    """

    # Perform the histogramming operation.
    out = data
    # histogrammed_data_1 = data.hist(gamma=gamma_binning["region_1"], two_theta=two_theta_binning)
    # histogrammed_data_2 = data.hist(gamma=gamma_binning["region_2"], two_theta=two_theta_binning)

    # out = sc.DataGroup({"region_1": histogrammed_data_1, "region_2": histogrammed_data_2})
    return View1D[RunType](out)
