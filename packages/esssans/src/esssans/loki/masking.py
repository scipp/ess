# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Masking functions for the loki workflow.
"""
from typing import NewType, Optional

import scipp as sc

from ..types import (
    CalibratedMaskedData,
    CleanMasked,
    DataWithLogicalDims,
    MaskedData,
    Numerator,
    RunType,
    SampleRun,
)

DetectorLowCountsStrawMask = NewType('DetectorLowCountsStrawMask', sc.Variable)
"""Detector low-counts straw mask"""
DetectorBadStrawsMask = NewType('DetectorBadStrawsMask', sc.Variable)
"""Detector bad straws mask"""
DetectorBeamStopMask = NewType('DetectorBeamStopMask', sc.Variable)
"""Detector beam stop mask"""
DetectorTubeEdgeMask = NewType('DetectorTubeEdgeMask', sc.Variable)
"""Detector tube edge mask"""


def detector_straw_mask(
    sample_straws: CalibratedMaskedData[SampleRun],
) -> DetectorLowCountsStrawMask:
    dims = list(set(sample_straws.dims) - {'straw'})
    return DetectorLowCountsStrawMask(
        sample_straws.sum(dims).data < sc.scalar(300.0, unit='counts')
    )


def detector_beam_stop_mask(
    sample_straws: CalibratedMaskedData[SampleRun],
) -> DetectorBeamStopMask:
    pos = sample_straws.coords['position'].copy()
    pos.fields.z *= 0.0
    return DetectorBeamStopMask((sc.norm(pos) < sc.scalar(0.042, unit='m')))


def detector_tube_edge_mask(
    sample_straws: CalibratedMaskedData[SampleRun],
) -> DetectorTubeEdgeMask:
    return DetectorTubeEdgeMask(
        (abs(sample_straws.coords['position'].fields.x) > sc.scalar(0.36, unit='m'))
        | (abs(sample_straws.coords['position'].fields.y) > sc.scalar(0.28, unit='m'))
    )


def mask_detectors(
    da: DataWithLogicalDims[RunType],
) -> MaskedData[RunType]:
    """Apply pixel-specific masks to raw data.

    Parameters
    ----------
    da:
        Raw data.
    """
    # Beam stop
    da = da.copy(deep=False)
    counts = da.bins.sum().data
    r = sc.sqrt(
        da.coords['position'].fields.x ** 2 + da.coords['position'].fields.y ** 2
    )
    da.masks['low_counts_middle'] = (counts < sc.scalar(20.0, unit='counts')) & (
        r < sc.scalar(0.075, unit='m')
    )
    # Low counts
    da.masks['very_low_counts'] = counts < sc.scalar(3.0, unit='counts')
    return MaskedData[RunType](da)


def mask_after_calibration(
    da: CalibratedMaskedData[RunType],
    lowcounts_straw_mask: Optional[DetectorLowCountsStrawMask],
    beam_stop_mask: Optional[DetectorBeamStopMask],
    tube_edge_mask: Optional[DetectorTubeEdgeMask],
) -> CleanMasked[RunType, Numerator]:
    """Apply pixel-specific masks to raw data.

    Parameters
    ----------
    da:
        Raw data.
    lowcounts_straw_mask:
        Mask for straws with low counts.
    beam_stop_mask:
        Mask for beam stop.
    tube_edge_mask:
        Mask for tube edges.
    """
    da = da.copy(deep=False)
    # Clear masks from beam center finding step, as they are potentially using harsh
    # thresholds which could remove some of the interesting signal.
    da.masks.clear()
    if lowcounts_straw_mask is not None:
        da.masks['low_counts'] = lowcounts_straw_mask
    if beam_stop_mask is not None:
        da.masks['beam_stop'] = beam_stop_mask
    if tube_edge_mask is not None:
        da.masks['tube_edges'] = tube_edge_mask
    return CleanMasked[RunType, Numerator](da)


providers = (
    detector_straw_mask,
    detector_beam_stop_mask,
    detector_tube_edge_mask,
    mask_detectors,
    mask_after_calibration,
)
