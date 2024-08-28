# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import sys
from pathlib import Path

import scipp as sc
from ess import loki
from ess.loki import LokiAtLarmorWorkflow
from ess.reduce import workflow

from ess.sans.types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    BeamCenter,
    Filename,
    IofQ,
    PixelMaskFilename,
    QBins,
    ReturnEvents,
    SampleRun,
    UncertaintyBroadcastMode,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_params


def test_sans_workflow_registers_subclasses():
    # Because it was imported
    assert LokiAtLarmorWorkflow in workflow.workflow_registry
    count = len(workflow.workflow_registry)

    @workflow.register_workflow
    class MyWorkflow: ...

    assert MyWorkflow in workflow.workflow_registry
    assert len(workflow.workflow_registry) == count + 1


def test_loki_workflow_parameters_returns_filtered_params():
    wf = LokiAtLarmorWorkflow()
    parameters = workflow.get_parameters(wf, (IofQ[SampleRun],))
    assert Filename[SampleRun] in parameters
    assert Filename[BackgroundRun] not in parameters


def test_loki_workflow_parameters_returns_no_params_for_no_outputs():
    wf = LokiAtLarmorWorkflow()
    parameters = workflow.get_parameters(wf, ())
    assert not parameters


def test_loki_workflow_parameters_with_param_returns_param():
    wf = LokiAtLarmorWorkflow()
    parameters = workflow.get_parameters(wf, (ReturnEvents,))
    assert parameters.keys() == {ReturnEvents}


def test_loki_workflow_compute_with_single_pixel_mask():
    params = make_params(no_masks=False)
    params[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.drop
    wf = LokiAtLarmorWorkflow()
    for key, val in params.items():
        wf[key] = val
    wf[PixelMaskFilename] = loki.data.loki_tutorial_mask_filenames()[0]
    # For simplicity, insert a fake beam center instead of computing it.
    wf[BeamCenter] = sc.vector([0.0, 0.0, 0.0], unit='m')

    result = wf.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)
    assert sc.identical(result.coords['Q'], params[QBins])
    assert result.sizes['Q'] == 100
