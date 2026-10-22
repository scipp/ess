# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""POWTEX texture reduction workflow."""

import sciline
import scipp as sc

from ..powtex.data import load_powtex_calibration, load_powtex_csv
from .conversion import project_onto_cylindrical_surface
from .correction import (
    normalize_by_vanadium,
    subtract_empty_can_from_sample,
    subtract_empty_can_from_vanadium,
)
from .grouping import (
    export_data,
    extract_rings_statistics,
    generate_dspacing_hist,
    generate_integrated_two_theta_hist,
    generate_two_theta_hist,
)
from .histogramming import (
    extract_detector_view_data,
    extract_flat_view_data,
    extract_gamma_slices_view_data,
    extract_gamma_slices_view_raw_data,
    extract_two_theta_rings_view_raw_data,
    extract_two_theta_rings_view_data,
    generate_gamma_binning_data,
    generate_two_theta_binning_data,
    integrate_dspacing,
    hist_2d,
    hist_3d,
)
from .types import BinningSettings


def default_parameters() -> dict:
    """Return default parameters for the POWTEX texture workflow."""
    return {
        BinningSettings: BinningSettings(
            gamma_cut_start=sc.scalar(225.0, unit="deg"), # voxel edge
            gamma_cut_end=sc.scalar(315.0, unit="deg"), # voxel edge
            drop_partial_bins=True,
            two_theta_min=sc.scalar(45.0, unit="deg"),
            two_theta_max=sc.scalar(135.0, unit="deg"),
        ),
    }


def TextureReduce() -> sciline.Pipeline:
    """Workflow with default parameters for POWTEX texture reduction."""
    return sciline.Pipeline(
        providers=(
            load_powtex_csv,
            load_powtex_calibration,
            project_onto_cylindrical_surface,
            generate_gamma_binning_data,
            generate_two_theta_binning_data,
            hist_3d,
            hist_2d,
            extract_detector_view_data,
            extract_flat_view_data,
            extract_gamma_slices_view_data,
            extract_gamma_slices_view_raw_data,
            extract_rings_statistics,
            extract_two_theta_rings_view_data,
            extract_two_theta_rings_view_raw_data,
            integrate_dspacing,
            subtract_empty_can_from_sample,
            subtract_empty_can_from_vanadium,
            normalize_by_vanadium,
            generate_dspacing_hist,
            generate_two_theta_hist,
            generate_integrated_two_theta_hist,
            export_data,
        ),
        params=default_parameters(),
    )


__all__ = [
    "TextureReduce",
]
