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
    DirectBeam,
    Filename,
    IofQ,
    PixelMaskFilename,
    QBins,
    ReturnEvents,
    SampleRun,
    UncertaintyBroadcastMode,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_workflow


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
    wf = make_workflow(no_masks=False)
    wf[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.drop
    wf[PixelMaskFilename] = loki.data.loki_tutorial_mask_filenames()[0]
    # For simplicity, insert a fake beam center instead of computing it.
    wf[BeamCenter] = sc.vector([0.0, 0.0, 0.0], unit='m')

    result = wf.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)
    assert sc.identical(result.coords['Q'], wf.compute(QBins))
    assert result.sizes['Q'] == 100


def test_loki_workflow_widget():
    from ess.reduce import ui

    results = {}
    widget = ui.workflow_widget(result_registry=results)
    # Select tutorial workflow
    select = widget.children[0].children[0]
    keys, values = zip(*select.options, strict=True)
    ind = keys.index('LokiAtLarmorTutorialWorkflow')
    select.value = values[ind]
    # Select IofQ[SampleRun] output
    wfw = widget.children[1].children[0]
    outputs = wfw.output_selection_box.typical_outputs_widget
    keys, values = zip(*outputs.options, strict=True)
    ind = keys.index('IofQ[SampleRun]')
    outputs.value = (values[ind],)
    # Refresh parameters
    pbox = wfw.parameter_box
    pbox.parameter_refresh_button.click()
    # Enable DirectBeam input
    pbox._input_widgets[DirectBeam].children[0].enabled = True
    pbox._input_widgets[DirectBeam].children[0].wrapped._option_box.value = None
    # Run the workflow
    rbox = wfw.result_box
    rbox.run_button.click()
    # Inspect results
    (da,) = results.values()
    assert da.dims == ('Q',)
    assert da.sizes['Q'] == pbox._input_widgets[QBins].children[0].fields['nbins'].value
