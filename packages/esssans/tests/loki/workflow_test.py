# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)


import pytest
import scipp as sc

from ess import loki
from ess.loki import LokiAtLarmorWorkflow
from ess.reduce import workflow
from ess.sans.types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    BeamCenter,
    Filename,
    IntensityQ,
    NeXusDetectorName,
    PixelMaskFilename,
    QBins,
    ReturnEvents,
    SampleRun,
    TimeOfFlightLookupTableFilename
    TofDetector,
    UncertaintyBroadcastMode,
)


def test_sans_workflow_registers_subclasses():
    # Because it was imported
    assert LokiAtLarmorWorkflow in workflow.workflow_registry
    count = len(workflow.workflow_registry)

    @workflow.register_workflow
    class MyWorkflow: ...

    assert MyWorkflow in workflow.workflow_registry
    assert len(workflow.workflow_registry) == count + 1


def test_loki_larmor_workflow_parameters_returns_filtered_params():
    wf = LokiAtLarmorWorkflow()
    parameters = workflow.get_parameters(wf, (IntensityQ[SampleRun],))
    assert Filename[SampleRun] in parameters
    assert Filename[BackgroundRun] not in parameters


def test_loki_larmor_workflow_parameters_returns_no_params_for_no_outputs():
    wf = LokiAtLarmorWorkflow()
    parameters = workflow.get_parameters(wf, ())
    assert not parameters


def test_loki_larmor_workflow_parameters_with_param_returns_param():
    wf = LokiAtLarmorWorkflow()
    parameters = workflow.get_parameters(wf, (ReturnEvents,))
    assert parameters.keys() == {ReturnEvents}


def test_loki_larmor_workflow_compute_with_single_pixel_mask(larmor_workflow):
    wf = larmor_workflow(no_masks=False)
    wf[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.drop
    wf[PixelMaskFilename] = loki.data.loki_tutorial_mask_filenames()[0]
    # For simplicity, insert a fake beam center instead of computing it.
    wf[BeamCenter] = sc.vector([0.0, 0.0, 0.0], unit='m')

    result = wf.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)
    assert sc.identical(result.coords['Q'], wf.compute(QBins))
    assert result.sizes['Q'] == 100


@pytest.mark.parametrize("bank", list(range(9)))
def test_loki_workflow_needs_tof_lookup_table(loki_workflow, bank):
    wf = loki_workflow()
    # For simplicity, insert a fake beam center instead of computing it.
    wf[BeamCenter] = sc.vector([0.0, 0.0, 0.0], unit='m')
    wf[NeXusDetectorName] = f'loki_detector_{bank}'
    with pytest.raises(KeyError, match='tof_lookup_table'):
        wf.compute(TofDetector[SampleRun])


@pytest.mark.parametrize("bank", list(range(9)))
def test_loki_workflow_can_compute_tof(loki_workflow, bank):
    wf = loki_workflow()
    # For simplicity, insert a fake beam center instead of computing it.
    wf[BeamCenter] = sc.vector([0.0, 0.0, 0.0], unit='m')
    wf[NeXusDetectorName] = f'loki_detector_{bank}'
    wf[TimeOfFlightLookupTableFilename] = loki.data.loki_tof_lookup_table_no_choppers()
    result = wf.compute(TofDetector[SampleRun])
    assert 'tof' in result.bins.coords

@pytest.mark.parametrize("bank", list(range(9)))
def test_loki_workflow_can_compute_iofq(loki_workflow, bank):
    wf = loki_workflow()
    # For simplicity, insert a fake beam center instead of computing it.
    wf[BeamCenter] = sc.vector([0.0, 0.0, 0.0], unit='m')
    wf[NeXusDetectorName] = f'loki_detector_{bank}'
    wf[TimeOfFlightLookupTableFilename] = loki.data.loki_tof_lookup_table_no_choppers()

    result = wf.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)
    assert sc.identical(result.coords['Q'], wf.compute(QBins))
