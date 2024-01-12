# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""
Masking functions for the loki workflow.
"""
from typing import NewType, Optional

import numpy as np
import scipp as sc

from ..types import (
    BeamStopPosition,
    BeamStopRadius,
    MaskedData,
    RawData,
    RunType,
    SampleRun,
)

DetectorLowCountsStrawMask = NewType('DetectorLowCountsStrawMask', sc.Variable)
"""Detector low-counts straw mask"""
DetectorBeamStopMask = NewType('DetectorBeamStopMask', sc.Variable)
"""Detector beam stop mask"""
DetectorTubeEdgeMask = NewType('DetectorTubeEdgeMask', sc.Variable)
"""Detector tube edge mask"""


def detector_straw_mask(
    sample_straws: RawData[SampleRun],
) -> DetectorLowCountsStrawMask:
    return DetectorLowCountsStrawMask(
        sample_straws.sum('pixel').data
        < sc.scalar(sample_straws.sizes['pixel'] * 0.5, unit='counts')
    )


def detector_beam_stop_mask(
    sample_straws: RawData[SampleRun],
    beam_stop_position: BeamStopPosition,
    beam_stop_radius: BeamStopRadius,
) -> DetectorBeamStopMask:
    pos = sample_straws.coords['position'] - beam_stop_position
    pos.fields.z *= 0.0
    return DetectorBeamStopMask((sc.norm(pos) < beam_stop_radius))


def detector_tube_edge_mask(
    sample_straws: RawData[SampleRun],
) -> DetectorTubeEdgeMask:
    other_dims = set(sample_straws.dims) - {'pixel'}
    size = np.prod([sample_straws.sizes[dim] for dim in other_dims])
    return DetectorTubeEdgeMask(
        sample_straws.sum(other_dims).data < sc.scalar(size * 0.5, unit='counts')
    )


def mask_detectors(
    da: RawData[RunType],
    lowcounts_straw_mask: Optional[DetectorLowCountsStrawMask],
    beam_stop_mask: Optional[DetectorBeamStopMask],
    tube_edge_mask: Optional[DetectorTubeEdgeMask],
) -> MaskedData[RunType]:
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
    if lowcounts_straw_mask is not None:
        da.masks['low_counts'] = lowcounts_straw_mask
    if beam_stop_mask is not None:
        da.masks['beam_stop'] = beam_stop_mask
    if tube_edge_mask is not None:
        da.masks['tube_edges'] = tube_edge_mask
    return MaskedData[RunType](da)


providers = (
    detector_straw_mask,
    detector_beam_stop_mask,
    detector_tube_edge_mask,
    mask_detectors,
)
