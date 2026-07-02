from typing import cast
import scipp as sc

from .correction import apply_lorentz_correction
from .masking import apply_masks, apply_masks_mlz, apply_masks_p2d
from .types import (
    CorrectedDetector,
    DspacingDetector,
    MaskedDetectorIDs,
    WavelengthDetector,
    RunType,
    TofMask,
    TwoThetaMask,
    WavelengthMask,
    DSpacingMask,
    DperpMask,
    MaskingData,
    NormalizedIntensity2D,
    ExtendedIntensity2D,
    MaskedExtendedIntensity2D
)

from .conversion import lambda_from_d_dperp


def add_masks_and_corrections(
    da: WavelengthDetector[RunType],
    masked_pixel_ids: MaskedDetectorIDs,
    tof_mask_func: TofMask,
    wavelength_mask_func: WavelengthMask,
    two_theta_mask_func: TwoThetaMask,
) -> CorrectedDetector[RunType]:
    masked = apply_masks(
        data=da,
        masked_pixel_ids=masked_pixel_ids,
        tof_mask_func=tof_mask_func,
        wavelength_mask_func=wavelength_mask_func,
        two_theta_mask_func=two_theta_mask_func,
    )
    out = apply_lorentz_correction(masked)
    return CorrectedDetector[RunType](out)


def add_masks_mlz(
    da: WavelengthDetector[RunType],
    tof_mask_func: MaskingData[TofMask],
    wavelength_mask_func: MaskingData[WavelengthMask],
    two_theta_mask_func: MaskingData[TwoThetaMask],
) -> CorrectedDetector[RunType]:
    if da is None:
        return cast(CorrectedDetector[RunType], None)
    masked = apply_masks_mlz(
        data=da,
        tof_mask_func=tof_mask_func,
        wavelength_mask_func=wavelength_mask_func,
        two_theta_mask_func=two_theta_mask_func,
    )
    # out = apply_lorentz_correction(masked) <- seems not to be included in Mantid workflow
    del da
    del tof_mask_func
    del wavelength_mask_func
    del two_theta_mask_func
    return CorrectedDetector[RunType](masked)


def append_two_theta_lambda(p2d_data_array: NormalizedIntensity2D) -> ExtendedIntensity2D:
    lambda_K = sc.scalar(1.0, unit="angstrom")
    d_centers = sc.midpoints(p2d_data_array.coords["dspacing"])
    dp_centers = sc.midpoints(p2d_data_array.coords["dperp"])

    # create lambda array in (d, dperp) space
    lam_2d = lambda_from_d_dperp(d_centers, dp_centers, lambda_K)
    lam_2d = lam_2d.transpose() # transpose to match the shape of p2d_data_array

    # create d and dperp arrays in (d, dperp) space
    dspacing_2d = sc.broadcast(d_centers, dims=["dperp", "dspacing"], shape=[dp_centers.size, d_centers.size])
    dperp_2d = sc.broadcast(dp_centers, dims=["dperp", "dspacing"], shape=[dp_centers.size, d_centers.size])

    # create two_theta array in (d, dperp) space
    two_theta_2d = (2.0 * sc.asin(0.5 * lam_2d / dspacing_2d)).to(unit="deg")

    # combine all the elements in a single Dataset
    ds = sc.Dataset(
        data={
            "lambda": lam_2d.data if isinstance(lam_2d, sc.DataArray) else lam_2d,
            "two_theta": two_theta_2d.data if isinstance(two_theta_2d, sc.DataArray) else two_theta_2d,
            "dspacing_2d": dspacing_2d.data if isinstance(dspacing_2d, sc.DataArray) else dspacing_2d,
            "dperp_2d": dperp_2d.data if isinstance(dperp_2d, sc.DataArray) else dperp_2d,
            "intensity": p2d_data_array.data.transpose() if isinstance(p2d_data_array, sc.DataArray) else p2d_data_array,
        },
        coords={
            "dperp": p2d_data_array.coords["dperp"],
            "dspacing": p2d_data_array.coords["dspacing"],
        }
    )
    del p2d_data_array
    return ExtendedIntensity2D(ds)


def add_masks_p2d_data(
    data: ExtendedIntensity2D,
    wavelength_mask_func: MaskingData[WavelengthMask],
    two_theta_mask_func: MaskingData[TwoThetaMask],
    dspacing_mask_func: MaskingData[DSpacingMask],
    dperp_mask_func: MaskingData[DperpMask],
) -> MaskedExtendedIntensity2D:

    masked = apply_masks_p2d(
        data,
        wavelength_mask_func,
        two_theta_mask_func,
        dspacing_mask_func,
        dperp_mask_func
    )
    del data
    return MaskedExtendedIntensity2D(masked)




providers = (add_masks_and_corrections,)
