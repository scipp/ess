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
    """
    This mask aims to remove straws with low counts. It is based on the assumption
    that the integrated counts in a straw should be at least greater than the number of
    pixels in the straw (within a fudge factor of 0.5).
    This is not true for straws that are not hit by the beam.
    Because the straws are horizontal, this mask will typically mask straws that are
    at the top and bottom edges of the detector panel.
    Note that this has only been tested on the main rear detector panel.
    """
    return DetectorLowCountsStrawMask(
        sample_straws.sum('pixel').data
        < sc.scalar(sample_straws.sizes['pixel'] * 0.5, unit='counts')
    )


def detector_beam_stop_mask(
    sample_straws: RawData[SampleRun],
    beam_stop_position: BeamStopPosition,
    beam_stop_radius: BeamStopRadius,
) -> DetectorBeamStopMask:
    """
    This mask aims to remove the beam stop. It masks a circular region around the beam.
    Usually, at this point in the workflow, the position of the beam is not accurately
    know (the beam center finder has not been called yet, because it requires masked
    data as an input). Therefore, only a rough estimate of the beam position is used.
    Note that this has only been tested on the main rear detector panel.
    """
    pos = sample_straws.coords['position'] - beam_stop_position
    pos.fields.z *= 0.0
    return DetectorBeamStopMask((sc.norm(pos) < beam_stop_radius))


def detector_tube_edge_mask(
    sample_straws: RawData[SampleRun],
) -> DetectorTubeEdgeMask:
    """
    This mask aims to remove the left and right edges of the detector panel.
    Those areas typically have low counts. This mask is based on the assumption
    that the counts integrated inside a vertical slab (summed over the ``straw`` and
    ``layer`` dimensions) should be at least greater that the number of pixels that
    make up the slab (within a fudge factor of 0.5).
    This is not true for slabs that are not hit by the beam.
    Note that this has only been tested on the main rear detector panel.
    """
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
