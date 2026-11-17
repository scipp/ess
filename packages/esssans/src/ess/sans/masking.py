# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Masking functions for the loki workflow.
"""

from ess.reduce.masking import read_pixel_masks, to_detector_masks

from .types import (
    CorrectedDetector,
    DetectorBankSizes,
    DetectorMasks,
    EmptyDetector,
    NeXusDetectorName,
    Numerator,
    PixelMasks,
    RunType,
    SampleRun,
    WavelengthDetector,
)


def to_detector_mask(
    detector: EmptyDetector[SampleRun],
    masks: PixelMasks,
    bank: NeXusDetectorName,
    bank_sizes: DetectorBankSizes,
) -> DetectorMasks:
    """Create detector masks for a bank from the pixel masks of the instrument.

    Parameters
    ----------
    detector:
        Detector of the sample run, which gives the logical dimensions and detector IDs
        of the bank.
    masks:
        Pixel masks of the instrument.
    bank:
        Name of the detector bank.
    bank_sizes:
        Sizes of the banks of the instrument, which define the valid bank names.
    """
    return DetectorMasks(
        to_detector_masks(masks, detector, bank=bank, known_banks=bank_sizes)
    )


def apply_pixel_masks(
    data: WavelengthDetector[RunType],
    masks: DetectorMasks,
) -> CorrectedDetector[RunType, Numerator]:
    """Apply pixel-specific masks to raw data.

    Parameters
    ----------
    data:
        Raw data with configured component positions.
    masks:
        A series of masks.
    """
    return CorrectedDetector[RunType, Numerator](data.assign_masks(masks))


providers = (
    read_pixel_masks,
    to_detector_mask,
    apply_pixel_masks,
)
