# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""POWTEX powder reduction workflows."""

from typing import cast

import sciline
import scipp as sc

from .conversion import lambda_from_d_dperp
from .data import (
    export_to_p2d,
    extract_masked_intensity,
    load_powgen_calibration_mlz,
    load_powgen_dperp_mask_mlz,
    load_powgen_dspacing_mask_mlz,
    load_powgen_mlz,
    load_powgen_tof_mask_mlz,
    load_powgen_two_theta_mask_mlz,
    load_powgen_wavelength_mask_mlz,
    prepare_for_p2d_export,
)
from .conversion import (
    add_scattering_coordinates_from_positions_mlz,
    p2d_coordinate_transformation_graph_mlz,
)
from .correction import (
    handle_negative_intensity_sample,
    handle_negative_intensity_vanadium,
    normalize_by_vanadium_mlz,
    subtract_sample_empty_can_mlz,
    subtract_vanadium_empty_can_mlz,
)
from .filtering import flatten_to_spectra
from .grouping import focus_to_dspacing_dperp
from .masking import apply_masks_mlz, apply_masks_p2d
from .types import (
    CalibrationRun,
    CorrectedDetector,
    DSpacingMask,
    DperpBins,
    DperpMask,
    DspacingBins,
    EmptyCanRun,
    EmptyVanadiumCanRun,
    ExtendedIntensity2D,
    FilenameMLZ,
    FilepathMLZ,
    KeepEventsMLZ,
    MaskedExtendedIntensity2D,
    MaskedIntensity2D,
    MaskingData,
    NegativeIntensityResetValue,
    NormalizedIntensity2D,
    P2DOutputFilename,
    RunType,
    SampleRun,
    ShiftNegativeIntensity,
    TofMask,
    TwoThetaMask,
    UncertaintyBroadcastMode,
    VanadiumRun,
    WavelengthDetector,
    WavelengthMask,
)


def add_masks_mlz(
    da: WavelengthDetector[RunType],
    tof_mask_func: TofMask,
    wavelength_mask_func: WavelengthMask,
    two_theta_mask_func: TwoThetaMask,
) -> CorrectedDetector[RunType]:
    """Apply MLZ event masks without Lorentz correction."""
    if da is None:
        return cast(CorrectedDetector[RunType], None)
    masked = apply_masks_mlz(
        data=da,
        tof_mask_func=tof_mask_func,
        wavelength_mask_func=wavelength_mask_func,
        two_theta_mask_func=two_theta_mask_func,
    )
    return CorrectedDetector[RunType](masked)


def append_two_theta_lambda(data: NormalizedIntensity2D) -> ExtendedIntensity2D:
    """Append wavelength and two-theta grids to normalized P2D intensity."""
    lambda_k = sc.scalar(1.0, unit="angstrom")
    d_centers = sc.midpoints(data.coords["dspacing"])
    dp_centers = sc.midpoints(data.coords["dperp"])

    lam_2d = lambda_from_d_dperp(d_centers, dp_centers, lambda_k).transpose()
    dspacing_2d = sc.broadcast(
        d_centers,
        dims=["dperp", "dspacing"],
        shape=[dp_centers.size, d_centers.size],
    )
    dperp_2d = sc.broadcast(
        dp_centers,
        dims=["dperp", "dspacing"],
        shape=[dp_centers.size, d_centers.size],
    )
    two_theta_2d = (2.0 * sc.asin(0.5 * lam_2d / dspacing_2d)).to(unit="deg")

    ds = sc.Dataset(
        data={
            "lambda": lam_2d.data if isinstance(lam_2d, sc.DataArray) else lam_2d,
            "two_theta": (
                two_theta_2d.data
                if isinstance(two_theta_2d, sc.DataArray)
                else two_theta_2d
            ),
            "dspacing_2d": (
                dspacing_2d.data
                if isinstance(dspacing_2d, sc.DataArray)
                else dspacing_2d
            ),
            "dperp_2d": (
                dperp_2d.data if isinstance(dperp_2d, sc.DataArray) else dperp_2d
            ),
            "intensity": data.data.transpose()
            if isinstance(data, sc.DataArray)
            else data,
        },
        coords={
            "dperp": data.coords["dperp"],
            "dspacing": data.coords["dspacing"],
        },
    )
    return ExtendedIntensity2D(ds)


def add_masks_p2d_data(
    data: ExtendedIntensity2D,
    wavelength_mask_func: MaskingData[WavelengthMask],
    two_theta_mask_func: MaskingData[TwoThetaMask],
    dspacing_mask_func: MaskingData[DSpacingMask],
    dperp_mask_func: MaskingData[DperpMask],
) -> MaskedExtendedIntensity2D:
    """Apply masks to the extended P2D dataset."""
    return MaskedExtendedIntensity2D(
        apply_masks_p2d(
            data,
            wavelength_mask_func,
            two_theta_mask_func,
            dspacing_mask_func,
            dperp_mask_func,
        )
    )


def default_parameters() -> dict:
    """Return default parameters for the POWTEX P2D workflow."""
    data_path = "/home/koshchii/Instruments/POWTEX/Scipp/data/yannick/converted"
    return {
        FilepathMLZ[SampleRun]: data_path,
        FilepathMLZ[VanadiumRun]: data_path,
        FilepathMLZ[EmptyCanRun]: data_path,
        FilepathMLZ[CalibrationRun]: data_path,
        FilepathMLZ[EmptyVanadiumCanRun]: data_path,
        FilenameMLZ[SampleRun]: None,
        FilenameMLZ[VanadiumRun]: None,
        FilenameMLZ[EmptyCanRun]: None,
        FilenameMLZ[CalibrationRun]: None,
        FilenameMLZ[EmptyVanadiumCanRun]: None,
        DspacingBins: sc.linspace("dspacing", 0.4, 2.0, 321, unit="angstrom"),
        DperpBins: sc.linspace("dperp", 0.2, 4.0, 381, unit="angstrom"),
        KeepEventsMLZ: False,
        ShiftNegativeIntensity[SampleRun]: True,
        ShiftNegativeIntensity[VanadiumRun]: False,
        NegativeIntensityResetValue[SampleRun]: 0.0,
        NegativeIntensityResetValue[VanadiumRun]: 0.0,
        UncertaintyBroadcastMode: UncertaintyBroadcastMode.drop,
        P2DOutputFilename: (
            "/home/koshchii/Instruments/POWTEX/Scipp/data/output/reduced.p2d"
        ),
    }


def PowderReduceP2D() -> sciline.Pipeline:
    """Workflow with default parameters for POWTEX P2D powder reduction."""
    return sciline.Pipeline(
        providers=(
            add_scattering_coordinates_from_positions_mlz,
            # add_masks_mlz, <- might be better to add masks in this step in future, because masking binned data can bring additional distortion
            add_masks_p2d_data,
            append_two_theta_lambda,
            export_to_p2d,
            extract_masked_intensity,
            flatten_to_spectra,
            focus_to_dspacing_dperp,
            handle_negative_intensity_sample,
            handle_negative_intensity_vanadium,
            load_powgen_calibration_mlz,
            #load_powgen_dperp_mask_mlz,
            #load_powgen_dspacing_mask_mlz,
            load_powgen_mlz,
            #load_powgen_tof_mask_mlz,
            #load_powgen_two_theta_mask_mlz,
            #load_powgen_wavelength_mask_mlz,
            # extract_accumulated_proton_charge,
            # normalize_by_proton_charge_mlz, <- this and above seem to be absent in PowderReduceP2D
            normalize_by_vanadium_mlz,
            p2d_coordinate_transformation_graph_mlz,
            prepare_for_p2d_export,
            subtract_sample_empty_can_mlz,
            subtract_vanadium_empty_can_mlz,
        ),
        params=default_parameters(),
    )


providers = (
    add_masks_mlz,
    add_masks_p2d_data,
    append_two_theta_lambda,
)

__all__ = [
    "MaskedIntensity2D",
    "PowderReduceP2D",
    "add_masks_mlz",
    "add_masks_p2d_data",
    "append_two_theta_lambda",
    "default_parameters",
]
