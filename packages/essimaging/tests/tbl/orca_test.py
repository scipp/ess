# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pytest
import sciline as sl
import scipp as sc
import scippnexus as sx
from scipp.testing import assert_identical

import ess.tbl.data  # noqa: F401
from ess import tbl
from ess.imaging.types import (
    BackgroundSubtractedDetector,
    CorrectedDetector,
    DarkBackgroundRun,
    Filename,
    FluxNormalizedDetector,
    MaskingRules,
    NeXusDetectorName,
    NormalizedImage,
    OpenBeamRun,
    Position,
    ProtonCharge,
    RawDetector,
    SampleRun,
    UncertaintyBroadcastMode,
)
from ess.tbl import orca


@pytest.fixture
def workflow() -> sl.Pipeline:
    """
    Workflow for normalizing TBL Orca images.
    """

    wf = orca.OrcaNormalizedImagesWorkflow()
    wf[Filename[SampleRun]] = tbl.data.tbl_lego_sample_run()
    wf[Filename[DarkBackgroundRun]] = tbl.data.tbl_lego_dark_run()
    wf[Filename[OpenBeamRun]] = tbl.data.tbl_lego_openbeam_run()
    wf[MaskingRules] = {}
    wf[NeXusDetectorName] = 'orca_detector'
    wf[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    wf[Position[sx.NXsample, SampleRun]] = sc.vector([0.0, 0.0, 0.0], unit='m')
    return wf


@pytest.mark.parametrize("run", [SampleRun, OpenBeamRun, DarkBackgroundRun])
def test_workflow_loads_raw_data(workflow, run):
    da = workflow.compute(RawDetector[run])
    assert "position" in da.coords
    assert "time" in da.coords
    assert da.ndim == 3
    assert "time" in da.dims


@pytest.mark.parametrize("run", [SampleRun, OpenBeamRun, DarkBackgroundRun])
def test_workflow_loads_proton_charge(workflow, run):
    pc = workflow.compute(ProtonCharge[run])
    assert "time" in pc.coords
    assert "time" in pc.dims
    assert pc.unit == "uC"


@pytest.mark.parametrize("run", [SampleRun, OpenBeamRun, DarkBackgroundRun])
def test_workflow_applies_masks(workflow, run):
    workflow[MaskingRules] = {
        'y_pixel_offset': lambda x: x > sc.scalar(0.082, unit='m')
    }
    da = workflow.compute(RawDetector[run])
    masked_da = workflow.compute(CorrectedDetector[run])
    assert 'y_pixel_offset' in masked_da.masks
    assert da.sum().value > masked_da.sum().value


@pytest.mark.parametrize("run", [OpenBeamRun, DarkBackgroundRun])
def test_workflow_normalizes_by_proton_charge(workflow, run):
    da = workflow.compute(FluxNormalizedDetector[run])
    # Dark and open beam runs have been averaged over time dimension
    assert da.ndim == 2
    assert "time" not in da.dims
    # TODO: should it be "counts / uC"?
    assert da.unit == "1 / uC"


def test_workflow_normalizes_sample_by_proton_charge(workflow):
    da = workflow.compute(FluxNormalizedDetector[SampleRun])
    # Sample run retains time dimension
    assert da.ndim == 3
    assert "time" in da.dims
    # TODO: should it be "counts / uC"?
    assert da.unit == "1 / uC"


@pytest.mark.parametrize("run", [OpenBeamRun, SampleRun])
def test_workflow_subtracts_dark_background(workflow, run):
    background = workflow.compute(FluxNormalizedDetector[DarkBackgroundRun])
    before = workflow.compute(FluxNormalizedDetector[run])
    after = workflow.compute(BackgroundSubtractedDetector[run])

    assert_identical(sc.values(after), sc.values(before) - sc.values(background))


def test_workflow_computes_normalized_image(workflow):
    sample = workflow.compute(BackgroundSubtractedDetector[SampleRun])
    open_beam = workflow.compute(BackgroundSubtractedDetector[OpenBeamRun])
    normalized = workflow.compute(NormalizedImage)

    assert_identical(sc.values(normalized), sc.values(sample) / sc.values(open_beam))
    assert normalized.unit == sc.units.dimensionless
