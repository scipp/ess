from typing import cast

from .correction import apply_lorentz_correction
from .masking import apply_masks, apply_masks_mlz
from .types import (
    CorrectedDetector,
    DspacingDetector,
    MaskedDetectorIDs,
    WavelengthDetector,
    RunType,
    TofMask,
    TwoThetaMask,
    WavelengthMask,
    MaskingData
)


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


providers = (add_masks_and_corrections,)
