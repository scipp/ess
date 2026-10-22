# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""Grouping helpers for POWTEX texture reduction."""

from .types import (
    DspacingHist,
    GammaSlicesViewData,
    GammaSliceWidth,
    IntegratedTwoThetaHist,
    NormalisedTexture2D,
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


def generate_dspacing_hist(
    gamma_bin_width: GammaSliceWidth,
    dummy_gamma_view: GammaSlicesViewData,
    dummy_theta_view: TwoThetaRingsViewData,
) -> DspacingHist:
    """Generate a d-spacing histogram placeholder."""
    _ = gamma_bin_width
    _ = dummy_theta_view
    return DspacingHist(dummy_gamma_view)


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
    data_1: DspacingHist,
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
    generate_dspacing_hist,
    generate_integrated_two_theta_hist,
    generate_two_theta_hist,
    view_1d_gamma_slices,
    view_2d_rings,
)
