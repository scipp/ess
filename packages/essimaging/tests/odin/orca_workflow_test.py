# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import ess.odin.data  # noqa: F401
import pytest
import sciline as sl
import scipp as sc
from ess import odin
from scipp.testing import assert_identical

from ess.imaging.types import (
    AllRuns,
    BackgroundSubtractedDetector,
    CorrectedDetector,
    DarkBackgroundRun,
    Filename,
    FluxNormalizedDetector,
    ImageKey,
    MaskingRules,
    MeanDarkFrame,
    NeXusDetectorName,
    NormalizedImage,
    OpenBeamRun,
    ProtonCharge,
    RawDetector,
    SampleRun,
    UncertaintyBroadcastMode,
)


@pytest.fixture
def workflow() -> sl.Pipeline:
    """
    Workflow for normalizing ODIN Orca images.
    """

    wf = odin.OdinOrcaWorkflow()
    wf[Filename[AllRuns]] = odin.data.odin_lego_images()
    wf[MaskingRules] = {}
    wf[NeXusDetectorName] = 'histogram_mode_detectors/orca'
    wf[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    return wf


def test_workflow_loads_raw_data(workflow):
    da = workflow.compute(RawDetector[AllRuns])
    assert "position" in da.coords
    assert "time" in da.coords
    assert da.ndim == 3
    assert "time" in da.dims


def test_workflow_loads_proton_charge(workflow):
    pc = workflow.compute(ProtonCharge[AllRuns])
    assert "time" in pc.coords
    assert "time" in pc.dims
    assert pc.unit == "uC"


def test_workflow_applies_masks(workflow):
    workflow[MaskingRules] = {
        'y_pixel_offset': lambda x: x > sc.scalar(0.082, unit='m')
    }
    da = workflow.compute(RawDetector[AllRuns])
    masked_da = workflow.compute(CorrectedDetector[AllRuns])
    assert 'y_pixel_offset' in masked_da.masks
    assert da.sum().value > masked_da.sum().value


def test_workflow_normalizes_by_proton_charge(workflow):
    da = workflow.compute(FluxNormalizedDetector[AllRuns])
    assert da.ndim == 3
    assert "time" in da.dims
    # TODO: should it be "counts / uC"?
    assert da.unit == "1 / uC"


def test_workflow_computes_image_key(workflow):
    image_key = workflow.compute(ImageKey)
    assert image_key.ndim == 1
    assert "time" in image_key.dims
    assert image_key.unit is None


@pytest.mark.parametrize("run", [OpenBeamRun, DarkBackgroundRun, SampleRun])
def test_workflow_extracts_different_runs_according_to_image_key(workflow, run):
    da = workflow.compute(FluxNormalizedDetector[run])
    assert da.ndim == 3
    assert "time" in da.dims
    # Data should be normalized by proton charge, so unit should be "1 / uC"
    assert da.unit == "1 / uC"


def test_workflow_computes_mean_dark_frame(workflow):
    dark_frames = workflow.compute(FluxNormalizedDetector[DarkBackgroundRun])
    mean_dark_frame = workflow.compute(MeanDarkFrame)
    assert mean_dark_frame.ndim == 2
    assert "time" not in mean_dark_frame.dims
    assert mean_dark_frame.unit == dark_frames.unit


@pytest.mark.parametrize("run", [OpenBeamRun, SampleRun])
def test_workflow_subtracts_dark_background(workflow, run):
    background = workflow.compute(MeanDarkFrame)
    before = workflow.compute(FluxNormalizedDetector[run])
    after = workflow.compute(BackgroundSubtractedDetector[run])

    assert_identical(sc.values(after), sc.values(before) - sc.values(background))


def test_workflow_computes_normalized_image(workflow):
    sample = workflow.compute(BackgroundSubtractedDetector[SampleRun])
    open_beam = workflow.compute(BackgroundSubtractedDetector[OpenBeamRun])
    normalized = workflow.compute(NormalizedImage)

    assert_identical(
        sc.values(normalized), sc.values(sample) / sc.values(open_beam.mean('time'))
    )
    assert normalized.unit == sc.units.dimensionless
