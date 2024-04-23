# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import NewType, Optional

import scipp as sc

from ..sans.types import MaskedData, SampleRun, ScatteringRunType, TofData

DetectorEdgeMask = NewType('DetectorEdgeMask', sc.Variable)
"""Detector edge mask"""

LowCountThreshold = NewType('LowCountThreshold', sc.Variable)
"""Threshold below which detector pixels should be masked
(low-counts on the edges of the detector panel, and the beam stop)"""

SampleHolderMask = NewType('SampleHolderMask', sc.Variable)
"""Sample holder mask"""


def detector_edge_mask(
    sample: TofData[SampleRun],
) -> DetectorEdgeMask:
    mask_edges = (
        sc.abs(sample.coords['position'].fields.x) > sc.scalar(0.48, unit='m')
    ) | (sc.abs(sample.coords['position'].fields.y) > sc.scalar(0.45, unit='m'))
    return DetectorEdgeMask(mask_edges)


def sample_holder_mask(
    sample: TofData[SampleRun],
    low_counts_threshold: LowCountThreshold,
) -> SampleHolderMask:
    summed = sample.hist()
    holder_mask = (
        (summed.data < low_counts_threshold)
        & (sample.coords['position'].fields.x > sc.scalar(0, unit='m'))
        & (sample.coords['position'].fields.x < sc.scalar(0.42, unit='m'))
        & (sample.coords['position'].fields.y < sc.scalar(0.05, unit='m'))
        & (sample.coords['position'].fields.y > sc.scalar(-0.15, unit='m'))
    )
    return SampleHolderMask(holder_mask)


def mask_detectors(
    da: TofData[ScatteringRunType],
    edge_mask: Optional[DetectorEdgeMask],
    holder_mask: Optional[SampleHolderMask],
) -> MaskedData[ScatteringRunType]:
    """Apply pixel-specific masks to raw data.

    Parameters
    ----------
    da:
        Raw data.
    edge_mask:
        Mask for detector edges.
    holder_mask:
        Mask for sample holder.
    """
    da = da.copy(deep=False)
    if edge_mask is not None:
        da.masks['edges'] = edge_mask
    if holder_mask is not None:
        da.masks['holder_mask'] = holder_mask
    return MaskedData[ScatteringRunType](da)


providers = (detector_edge_mask, sample_holder_mask, mask_detectors)
