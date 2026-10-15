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
    UncertaintyBroadcastMode,
    VanadiumRun,
)
from .conversion import (
    add_scattering_coordinates_from_positions,
    p2d_coordinate_transformation_graph,
)
from .correction import (
    handle_negative_intensity_sample,
    handle_negative_intensity_vanadium,
    normalize_by_vanadium,
    subtract_sample_empty_can,
    subtract_vanadium_empty_can,
)
from .data import (
    export_to_p2d,
    extract_masked_intensity,
    load_powgen,
    load_powgen_calibration,
    prepare_for_p2d_export,
)
from .grouping import (
    append_two_theta_lambda,
    flatten_to_spectra,
    focus_to_dspacing_dperp,
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
        UncertaintyBroadcastMode: UncertaintyBroadcastMode.drop,
    }


def PowderReduceP2D() -> sciline.Pipeline:
    """Workflow with default parameters for POWTEX P2D powder reduction."""
    return sciline.Pipeline(
        providers=(
            add_scattering_coordinates_from_positions,
            # it might be better to add masks to loaded event data
            # using powtex.masking.add_masks, because masking binned data
            # can bring additional distortion. In other words, we might
            # want to use in the future add_masks after file loading
            # instead of add_masks_p2d_data after histogramming.
            add_masks_p2d_data,
            append_two_theta_lambda,
            export_to_p2d,
            extract_masked_intensity,
            flatten_to_spectra,
            focus_to_dspacing_dperp,
            handle_negative_intensity_sample,
            handle_negative_intensity_vanadium,
            load_powgen_calibration,
            # load_powgen_dperp_mask_mlz,
            # load_powgen_dspacing_mask_mlz,
            load_powgen,
            # load_powgen_tof_mask_mlz,
            # load_powgen_two_theta_mask_mlz,
            # load_powgen_wavelength_mask_mlz,
            # extract_accumulated_proton_charge,
            # normalize_by_proton_charge_mlz, <- this and above seem to be absent
            # in PowderReduceP2D
            normalize_by_vanadium,
            p2d_coordinate_transformation_graph,
            prepare_for_p2d_export,
            subtract_sample_empty_can,
            subtract_vanadium_empty_can,
        ),
        params=default_parameters(),
    )


providers = ()

__all__ = [
    "PowderReduceP2D",
]
