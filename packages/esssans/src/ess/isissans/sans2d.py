# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import NewType

import sciline
import scipp as sc
from ess.sans import SansWorkflow
from ess.sans.parameters import typical_outputs
from ess.sans.types import DetectorMasks, EmptyDetector, SampleRun

from ess.reduce.workflow import register_workflow

from .general import default_parameters
from .io import load_tutorial_direct_beam, load_tutorial_run
from .mantidio import providers as mantid_providers

DetectorEdgeMask = NewType('DetectorEdgeMask', sc.Variable | None)
"""Detector edge mask"""

LowCountThreshold = NewType('LowCountThreshold', sc.Variable)
"""Threshold below which detector pixels should be masked
(low-counts on the edges of the detector panel, and the beam stop)"""

SampleHolderMask = NewType('SampleHolderMask', sc.Variable | None)
"""Sample holder mask"""


def detector_edge_mask(sample: EmptyDetector[SampleRun]) -> DetectorEdgeMask:
    pos = sample.coords['position']
    mask_edges = (sc.abs(pos.fields.x) > sc.scalar(0.48, unit='m')) | (
        sc.abs(pos.fields.y) > sc.scalar(0.45, unit='m')
    )
    return DetectorEdgeMask(mask_edges)


def sample_holder_mask(
    sample: EmptyDetector[SampleRun],
    low_counts_threshold: LowCountThreshold,
) -> SampleHolderMask:
    pos = sample.coords['position']
    summed = sample.hist()
    holder_mask = (
        (summed.data < low_counts_threshold)
        & (pos.fields.x > sc.scalar(0, unit='m'))
        & (pos.fields.x < sc.scalar(0.42, unit='m'))
        & (pos.fields.y < sc.scalar(0.05, unit='m'))
        & (pos.fields.y > sc.scalar(-0.15, unit='m'))
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


@register_workflow
def Sans2dWorkflow() -> sciline.Pipeline:
    """Create Sans2d workflow with default parameters."""
    from . import providers as isis_providers

    # Note that the actual NeXus loading in this workflow will not be used for the
    # ISIS files, the providers inserted below will replace those steps.
    workflow = SansWorkflow()
    for provider in isis_providers + mantid_providers + providers:
        workflow.insert(provider)
    for key, param in default_parameters().items():
        workflow[key] = param
    workflow.typical_outputs = typical_outputs
    return workflow


@register_workflow
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
