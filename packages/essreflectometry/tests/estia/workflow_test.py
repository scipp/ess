# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import inspect

from ess.reduce import workflow as reduce_workflow

from ess import estia
from ess.estia.corrections import RunNormalization
from ess.reflectometry.types import CorrectionsToApply, NeXusDetectorName


def test_estia_workflow_uses_estia_defaults():
    params = estia.workflow.default_parameters()

    assert params[NeXusDetectorName] == "multiblade_detector"
    assert params[CorrectionsToApply] == estia.corrections.default_corrections


def test_estia_workflows_have_expected_run_normalization_defaults():
    assert (
        inspect.signature(estia.EstiaMcStasWorkflow).parameters["run_norm"].default
        is RunNormalization.none
    )
    assert (
        inspect.signature(estia.EstiaWorkflow).parameters["run_norm"].default
        is RunNormalization.proton_charge
    )


def test_estia_workflow_registers_run_normalization_variants():
    for wf in (
        estia.EstiaMcStasUnnormalizedWorkflow,
        estia.EstiaMcStasMonitorHistogramWorkflow,
        estia.EstiaMcStasMonitorIntegratedWorkflow,
        estia.EstiaMcStasProtonChargeWorkflow,
        estia.EstiaUnnormalizedWorkflow,
        estia.EstiaMonitorHistogramWorkflow,
        estia.EstiaMonitorIntegratedWorkflow,
        estia.EstiaProtonChargeWorkflow,
    ):
        assert wf in reduce_workflow.workflow_registry
