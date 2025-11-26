from .correction import apply_lorentz_correction
from .masking import apply_masks
from .types import (
    CorrectedDetector,
    MaskedDetectorIDs,
    RunType,
    TofMask,
    TwoThetaMask,
    WavelengthDetector,
    WavelengthMask,
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


providers = (add_masks_and_corrections,)
