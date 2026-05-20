# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import inspect

from ess.reduce import workflow as reduce_workflow

from ess import freia
from ess.freia.corrections import RunNormalization
from ess.reflectometry.types import CorrectionsToApply, NeXusDetectorName


def test_freia_workflow_uses_freia_defaults():
    params = freia.workflow.default_parameters()

    assert params[NeXusDetectorName] == "multiblade_detector"
    assert params[CorrectionsToApply] == freia.corrections.default_corrections


def test_freia_workflows_have_expected_run_normalization_defaults():
    assert (
        inspect.signature(freia.FreiaMcStasWorkflow).parameters["run_norm"].default
        is RunNormalization.none
    )
    assert (
        inspect.signature(freia.FreiaWorkflow).parameters["run_norm"].default
        is RunNormalization.proton_charge
    )


def test_freia_workflow_registers_run_normalization_variants():
    for wf in (
        freia.FreiaMcStasUnnormalizedWorkflow,
        freia.FreiaMcStasMonitorHistogramWorkflow,
        freia.FreiaMcStasMonitorIntegratedWorkflow,
        freia.FreiaMcStasProtonChargeWorkflow,
        freia.FreiaUnnormalizedWorkflow,
        freia.FreiaMonitorHistogramWorkflow,
        freia.FreiaMonitorIntegratedWorkflow,
        freia.FreiaProtonChargeWorkflow,
    ):
        assert wf in reduce_workflow.workflow_registry
