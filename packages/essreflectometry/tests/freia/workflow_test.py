# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from ess.reduce import workflow as reduce_workflow

from ess import freia


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
