# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import NewType

import sciline
import scipp as sc
from ess.sans import providers as sans_providers
from ess.sans.types import DetectorMasks, RawDetector, SampleRun

from .data import load_tutorial_direct_beam, load_tutorial_run
from .general import default_parameters
from .mantidio import providers as mantid_providers

DetectorEdgeMask = NewType('DetectorEdgeMask', sc.Variable | None)
"""Detector edge mask"""

LowCountThreshold = NewType('LowCountThreshold', sc.Variable)
"""Threshold below which detector pixels should be masked
(low-counts on the edges of the detector panel, and the beam stop)"""

SampleHolderMask = NewType('SampleHolderMask', sc.Variable | None)
"""Sample holder mask"""


# It may make more sense to depend on CalibratedDetector here, but the current
# x and y limits are before setting the beam center, so we use RawDetector
def detector_edge_mask(sample: RawDetector[SampleRun]) -> DetectorEdgeMask:
    mask_edges = (
        sc.abs(sample.coords['position'].fields.x) > sc.scalar(0.48, unit='m')
    ) | (sc.abs(sample.coords['position'].fields.y) > sc.scalar(0.45, unit='m'))
    return DetectorEdgeMask(mask_edges)


# It may make more sense to depend on CalibratedDetector here, but the current
# x and y limits are before setting the beam center, so we use RawDetector
def sample_holder_mask(
    sample: RawDetector[SampleRun], low_counts_threshold: LowCountThreshold
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


def to_detector_masks(
    edge_mask: DetectorEdgeMask, holder_mask: SampleHolderMask
) -> DetectorMasks:
    """Gather detector masks.

    Unlike :py:func:`ess.sans.masking.to_detector_mask`, this function does not rely on
    mapping using a table of mask filenames. Instead it directly returns a dictionary
    if multiple masks.

    Parameters
    ----------
    edge_mask:
        Mask for detector edges.
    holder_mask:
        Mask for sample holder.
    """
    masks = {}
    if edge_mask is not None:
        masks['edges'] = edge_mask
    if holder_mask is not None:
        masks['holder_mask'] = holder_mask
    return DetectorMasks(masks)


providers = (detector_edge_mask, sample_holder_mask, to_detector_masks)


def Sans2dWorkflow() -> sciline.Pipeline:
    """Create Sans2d workflow with default parameters."""
    from . import providers as isis_providers

    params = default_parameters()
    sans2d_providers = sans_providers + isis_providers + mantid_providers + providers
    return sciline.Pipeline(providers=sans2d_providers, params=params)


def Sans2dTutorialWorkflow() -> sciline.Pipeline:
    """
    Create Sans2d tutorial workflow.

    Equivalent to :func:`Sans2dWorkflow`, but with loaders for tutorial data instead
    of Mantid-based loaders.
    """
    workflow = Sans2dWorkflow()
    workflow.insert(load_tutorial_run)
    workflow.insert(load_tutorial_direct_beam)
    return workflow
