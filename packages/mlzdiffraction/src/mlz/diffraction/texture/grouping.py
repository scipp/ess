# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Grouping helpers for POWTEX texture reduction."""

import scipp as sc
import numpy as np

from .types import (
    DspacingViewData,
    GammaSlicesViewData,
    GammaSliceWidth,
    IntegratedTwoThetaViewData,
    CalibratedTexture2D,
    CalibratedTexture3D,
    ProcessedData,
    StatisticsData,
    TwoThetaViewData,
    TwoThetaRingsViewData,
)


def extract_rings_statistics(data: CalibratedTexture2D) -> StatisticsData:
    """Return the gamma rings statistics data."""
    return StatisticsData(data)


def extract_dspacing_hist_view_data(
    data: CalibratedTexture3D
) -> DspacingViewData:
    """Generate a d-spacing histogram."""
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


def extract_two_theta_hist_view_data(
    data: CalibratedTexture3D
) -> TwoThetaViewData:
    """Generate a two-theta histogram."""
    histogrammed_2d_1 = data["region_1"].sum(dim="dspacing")
    histogrammed_2d_2 = data["region_2"].sum(dim="dspacing")

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

    two_theta_bin_edges = histogrammed_2d_1.coords["two_theta"]

    return TwoThetaViewData(
        sc.DataArray(
            counts,
            coords={
                "gamma": gamma_bin_edges,
                "two_theta": two_theta_bin_edges,
            },
        )
    )


def extract_integrated_two_theta_hist_view_data(
    data: CalibratedTexture3D
) -> IntegratedTwoThetaViewData:
    """Generate an integrated two-theta histogram."""
    integrated_1 = data["region_1"].sum(dim="dspacing").sum(dim="gamma")
    integrated_2 = data["region_2"].sum(dim="dspacing").sum(dim="gamma")
    return IntegratedTwoThetaViewData(integrated_1 + integrated_2)


def export_data(
    data_1: DspacingViewData,
    data_2: TwoThetaViewData,
    data_3: IntegratedTwoThetaViewData,
) -> ProcessedData:
    """Return processed texture data for export."""
    _ = data_2
    _ = data_3
    return ProcessedData(data_1)


providers = (
    export_data,
    extract_rings_statistics,
    extract_dspacing_hist_view_data,
    extract_integrated_two_theta_hist_view_data,
    extract_two_theta_hist_view_data,
)
