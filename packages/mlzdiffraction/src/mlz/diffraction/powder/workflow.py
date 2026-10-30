# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

"""POWTEX powder reduction workflows."""

import sciline
import scipp as sc

from ..powtex.types import (
    KeepEvents,
    NegativeIntensityResetValue,
    SampleRun,
    ShiftNegativeIntensity,
    VanadiumRun,
)
from .conversion import (
    flatten_to_spectra,
    transform_to_scattering_coordinates,
    p2d_coordinate_transformation_graph,
)
from .correction import (
    handle_negative_intensity_sample,
    handle_negative_intensity_vanadium,
    normalise_by_vanadium,
    subtract_sample_empty_can,
    subtract_vanadium_empty_can,
)
from .data import (
    export_to_p2d,
    load_powgen,
    load_powgen_calibration,
)
from .grouping import (
    extend_data,
    extract_intensity_2d,
    focus_to_dspacing_dperp,
    prepare_for_p2d_export,
)
from .masking import add_masks_p2d_data
from .types import (
    DperpBins,
    DspacingBins,
)


def default_parameters() -> dict:
    """Return default parameters for the POWTEX P2D workflow."""
    return {
        DspacingBins: sc.linspace("dspacing", 0.4, 2.0, 321, unit="angstrom"),
        DperpBins: sc.linspace("dperp", 0.2, 4.0, 381, unit="angstrom"),
        KeepEvents: False,
        ShiftNegativeIntensity[SampleRun]: True,
        ShiftNegativeIntensity[VanadiumRun]: False,
        NegativeIntensityResetValue[SampleRun]: 0.0,
        NegativeIntensityResetValue[VanadiumRun]: 0.0,
    }


def PowderReduceP2D() -> sciline.Pipeline:
    """Workflow with default parameters for POWTEX P2D powder reduction."""
    return sciline.Pipeline(
        providers=(
            # it might be better to add masks to loaded event data
            # using powtex.masking.add_masks, because masking binned data
            # can bring additional distortion. In other words, we might
            # want to use in the future add_masks after file loading
            # instead of add_masks_p2d_data after histogramming.
            add_masks_p2d_data,
            export_to_p2d,
            extend_data,
            extract_intensity_2d,
            flatten_to_spectra,
            focus_to_dspacing_dperp,
            handle_negative_intensity_sample,
            handle_negative_intensity_vanadium,
            load_powgen_calibration,
            load_powgen,
            normalise_by_vanadium,
            p2d_coordinate_transformation_graph,
            prepare_for_p2d_export,
            subtract_sample_empty_can,
            subtract_vanadium_empty_can,
            transform_to_scattering_coordinates,
        ),
        params=default_parameters(),
    )


providers = ()

__all__ = [
    "PowderReduceP2D",
]
