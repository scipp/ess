# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)


import pytest
import scipp as sc
from ess import loki
from ess.loki import LokiAtLarmorTutorialWorkflow, LokiAtLarmorWorkflow
from ess.sans.parameters import parameters as sans_parameters
from ess.sans.types import (
    BackgroundSubtractedIofQ,
    BeamCenter,
    DetectorMasks,
    Filename,
    LookupTableFilename,
    NeXusDetectorName,
    PixelMaskFilename,
    QBins,
    QxBins,
    ReturnEvents,
    SampleRun,
    UncertaintyBroadcastMode,
    WavelengthDetector,
)
from sciline import UnsatisfiedRequirement

from ess.reduce import workflow


def test_sans_workflow_registers_subclasses():
    # Because it was imported
    assert LokiAtLarmorWorkflow in workflow.workflow_registry
    count = len(workflow.workflow_registry)

    @workflow.register_workflow()
    class MyWorkflow: ...

    assert MyWorkflow in workflow.workflow_registry
    assert len(workflow.workflow_registry) == count + 1


def test_loki_larmor_workflow_registers_parameter_model():
    spec = workflow.workflow_registry.get(LokiAtLarmorWorkflow)
    assert spec.parameters is sans_parameters


def test_loki_larmor_workflow_applies_parameter_model():
    wf = LokiAtLarmorWorkflow()
    spec = workflow.workflow_registry.get(LokiAtLarmorWorkflow)
    params = workflow.get_parameters(wf, (ReturnEvents,), spec.parameters)
    wf = workflow.assign_parameter_values(wf, {ReturnEvents: True}, params)

    assert wf.compute(ReturnEvents)


def test_loki_larmor_workflow_parameters_are_selected_from_graph():
    wf = LokiAtLarmorWorkflow()
    spec = workflow.workflow_registry.get(LokiAtLarmorWorkflow)

    params = workflow.get_parameters(wf, (BackgroundSubtractedIofQ,), spec.parameters)

    assert Filename[SampleRun] in params
    assert PixelMaskFilename in params
    assert QBins in params
    assert QxBins not in params


def test_loki_larmor_tutorial_workflow_preserves_outputs_and_masks():
    wf = LokiAtLarmorTutorialWorkflow()
    spec = workflow.workflow_registry.get(LokiAtLarmorTutorialWorkflow)

    labels = [
        label for label, _ in workflow.get_typical_outputs(wf, spec.typical_outputs)
    ]
    masks = wf.compute(DetectorMasks)

    assert 'IntensityQ[SampleRun]' in labels
    assert tuple(masks) == tuple(map(str, loki.data.loki_tutorial_mask_filenames()))


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
def test_loki_workflow_needs_lookup_table(loki_workflow, bank):
    wf = loki_workflow()
    # For simplicity, insert a fake beam center instead of computing it.
    wf[BeamCenter] = sc.vector([0.0, 0.0, 0.0], unit='m')
    wf[NeXusDetectorName] = f'loki_detector_{bank}'
    with pytest.raises(UnsatisfiedRequirement, match='LookupTableFilename'):
        wf.compute(WavelengthDetector[SampleRun])


@pytest.mark.parametrize("bank", list(range(9)))
def test_loki_workflow_can_compute_wavelength(loki_workflow, bank):
    wf = loki_workflow()
    # For simplicity, insert a fake beam center instead of computing it.
    wf[BeamCenter] = sc.vector([0.0, 0.0, 0.0], unit='m')
    wf[NeXusDetectorName] = f'loki_detector_{bank}'
    wf[LookupTableFilename] = loki.data.loki_lookup_table_no_choppers()
    result = wf.compute(WavelengthDetector[SampleRun])
    assert 'wavelength' in result.bins.coords


@pytest.mark.parametrize("bank", list(range(9)))
def test_loki_workflow_can_compute_iofq(loki_workflow, bank):
    wf = loki_workflow()
    # For simplicity, insert a fake beam center instead of computing it.
    wf[BeamCenter] = sc.vector([0.0, 0.0, 0.0], unit='m')
    wf[NeXusDetectorName] = f'loki_detector_{bank}'
    wf[LookupTableFilename] = loki.data.loki_lookup_table_no_choppers()

    result = wf.compute(BackgroundSubtractedIofQ)
    assert result.dims == ('Q',)
    assert sc.identical(result.coords['Q'], wf.compute(QBins))
