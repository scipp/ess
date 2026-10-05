# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""POWTEX texture reduction workflow."""

import sciline
import scipp as sc


from ..powtex.types import (
    CalibrationRun,
    EmptyCanRun,
    EmptyVanadiumCanRun,
    FilepathMLZ,
    SampleRun,
    VanadiumRun,
)

from .conversion import project_onto_cylindrical_surface
from .correction import (
    normalize_by_vanadium,
    subtract_empty_can_from_sample,
    subtract_empty_can_from_vanadium,
)
from .grouping import (
    export_data,
    generate_dspacing_hist,
    generate_integrated_two_theta_hist,
    generate_two_theta_hist,
    view_1d_gamma_slices,
    view_2d_rings,
)
from .histogramming import (
    generate_gamma_binning_data,
    generate_two_theta_binning_data,
    hist_2d,
    hist_3d,
    extract_gamma_slices_view_data,
    extract_flat_view_data,
    extract_two_theta_rings_view_data,
    extract_detector_view_data,
)
from .types import BinningSettings
from ..powtex.data import load_powtex_calibration, load_powtex_csv


def default_parameters() -> dict:
    """Return default parameters for the POWTEX texture workflow."""
    data_path = "/home/koshchii/Instruments/POWTEX/Scipp/data/texture/csv"
    return {
        FilepathMLZ[SampleRun]: data_path,
        FilepathMLZ[VanadiumRun]: data_path,
        FilepathMLZ[EmptyCanRun]: data_path,
        FilepathMLZ[CalibrationRun]: data_path,
        FilepathMLZ[EmptyVanadiumCanRun]: data_path,
        BinningSettings: BinningSettings(
            gamma_cut_start=sc.scalar(225.0, unit="deg"),
            gamma_cut_end=sc.scalar(315.0, unit="deg"),
            drop_partial_bins=True,
            include_gamma_gap_bins=False,
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
            extract_two_theta_rings_view_data,
            extract_gamma_slices_view_data,
            subtract_empty_can_from_sample,
            subtract_empty_can_from_vanadium,
            normalize_by_vanadium,
            view_2d_rings,
            view_1d_gamma_slices,
            generate_dspacing_hist,
            generate_two_theta_hist,
            generate_integrated_two_theta_hist,
            export_data,
        ),
        params=default_parameters(),
    )


__all__ = ["TextureReduce", "default_parameters"]
