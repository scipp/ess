# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""POWTEX texture reduction workflow."""

import sciline
import scipp as sc

from ..powtex.data import load_powtex_calibration, load_geant4_simulation_csv
from .conversion import project_onto_cylindrical_surface
from .correction import (
    normalize_by_vanadium,
    subtract_empty_can_from_sample,
    subtract_empty_can_from_vanadium,
)
from .grouping import (
    export_data,
    extract_dspacing_hist_view_data,
    extract_integrated_two_theta_hist_view_data,
    extract_rings_statistics,
    extract_two_theta_hist_view_data,
)
from .histogramming import (
    extract_detector_view_data,
    extract_flat_view_data,
    extract_gamma_slices_view_data,
    extract_gamma_slices_view_raw_data,
    extract_raw_sample_data_2d,
    extract_two_theta_rings_view_raw_data,
    extract_two_theta_rings_view_data,
    generate_gamma_binning_data,
    generate_two_theta_binning_data,
    histogram_3d,
    integrate_dspacing,
)
from .types import InstrumentViewSettings


def default_parameters() -> dict:
    """Return default parameters for the POWTEX texture workflow."""
    return {
        InstrumentViewSettings: InstrumentViewSettings(
            drop_partial_bins=True,
            gamma_cut_start=sc.scalar(225.0, unit="deg"), # voxel edge
            gamma_cut_end=sc.scalar(315.0, unit="deg"), # voxel edge
            two_theta_min=sc.scalar(45.0, unit="deg"), # voxel edge
            two_theta_max=sc.scalar(135.0, unit="deg"), # voxel edge
        ),
    }


def TextureReduce() -> sciline.Pipeline:
    """Workflow with default parameters for POWTEX texture reduction."""
    return sciline.Pipeline(
        providers=(
            export_data,
            extract_detector_view_data,
            extract_dspacing_hist_view_data,
            extract_flat_view_data,
            extract_gamma_slices_view_data,
            extract_gamma_slices_view_raw_data,
            extract_integrated_two_theta_hist_view_data,
            extract_raw_sample_data_2d,
            extract_rings_statistics,
            extract_two_theta_hist_view_data,
            extract_two_theta_rings_view_data,
            extract_two_theta_rings_view_raw_data,
            generate_gamma_binning_data,
            generate_two_theta_binning_data,
            histogram_3d,
            integrate_dspacing,
            load_geant4_simulation_csv,
            load_powtex_calibration,
            normalize_by_vanadium,
            project_onto_cylindrical_surface,
            subtract_empty_can_from_sample,
            subtract_empty_can_from_vanadium,
        ),
        params=default_parameters(),
    )


__all__ = [
    "TextureReduce",
]
