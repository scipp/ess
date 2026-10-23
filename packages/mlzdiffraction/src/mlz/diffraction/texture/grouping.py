# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Grouping helpers for POWTEX texture reduction."""

import scipp as sc
import numpy as np

from .types import (
    DspacingViewData,
    GammaSlicesViewData,
    GammaSliceWidth,
    IntegratedTwoThetaHist,
    NormalisedTexture2D,
    NormalisedTexture3D,
    ProcessedData,
    StatisticsData,
    TwoThetaHist,
    TwoThetaRingsViewData,
)


def view_2d_rings(data: NormalisedTexture2D) -> TwoThetaRingsViewData:
    """Return the two-theta ring view."""
    return TwoThetaRingsViewData(data)


def view_1d_gamma_slices(data: NormalisedTexture2D) -> GammaSlicesViewData:
    """Return the gamma-slice view."""
    return GammaSlicesViewData(data)


def extract_rings_statistics(data: NormalisedTexture2D) -> StatisticsData:
    """Return the gamma rings statistics data."""
    return StatisticsData(data)


def extract_dspacing_hist_view_data(
    data: NormalisedTexture3D
) -> DspacingViewData:
    """Generate a d-spacing histogram placeholder."""
    histogrammed_2d_1 = data["region_1"].sum(dim="two_theta")
    histogrammed_2d_2 = data["region_2"].sum(dim="two_theta")

    gamma_bin_edges_1 = histogrammed_2d_1.coords["gamma"]
    # Below, we drop the last element on purpose, because we do
    # not need to have an edge at 360 deg, which is equivalent
    # the one at 0 deg.
    gamma_bin_edges_2 = histogrammed_2d_2.coords["gamma"][:-1]
    # Below, the order is changed on purpose, in order not to
    # deal with gap. No good way to implement it in ascending
    # order.
    gamma_bin_edges = sc.concat(
        [gamma_bin_edges_2, gamma_bin_edges_1],  dim="gamma"
    )

    counts_1 = histogrammed_2d_1.data
    counts_2 = histogrammed_2d_2.data
    counts = sc.concat([counts_2, counts_1], dim="gamma")

    dspacing_bin_edges = histogrammed_2d_1.coords["dspacing"]

    return DspacingViewData(
        sc.DataArray(
            counts,
            coords={
                "gamma": gamma_bin_edges,
                "dspacing": dspacing_bin_edges,
            },
        )
    )


def generate_two_theta_hist(
    gamma_bin_width: GammaSliceWidth,
    dummy_gamma_view: GammaSlicesViewData,
    dummy_theta_view: TwoThetaRingsViewData,
) -> TwoThetaHist:
    """Generate a two-theta histogram placeholder."""
    _ = gamma_bin_width
    _ = dummy_theta_view
    return TwoThetaHist(dummy_gamma_view)


def generate_integrated_two_theta_hist(
    gamma_bin_width: GammaSliceWidth,
    dummy_gamma_view: GammaSlicesViewData,
    dummy_theta_view: TwoThetaRingsViewData,
) -> IntegratedTwoThetaHist:
    """Generate an integrated two-theta histogram placeholder."""
    _ = gamma_bin_width
    _ = dummy_theta_view
    return IntegratedTwoThetaHist(dummy_gamma_view)


def export_data(
    data_1: DspacingViewData,
    data_2: TwoThetaHist,
    data_3: IntegratedTwoThetaHist,
) -> ProcessedData:
    """Return processed texture data for export."""
    _ = data_2
    _ = data_3
    return ProcessedData(data_1)


providers = (
    export_data,
    extract_rings_statistics,
    extract_dspacing_hist_view_data,
    generate_integrated_two_theta_hist,
    generate_two_theta_hist,
    view_1d_gamma_slices,
    view_2d_rings,
)
