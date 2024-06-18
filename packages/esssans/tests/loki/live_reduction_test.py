# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import sciline as sl
from ess.loki.workflow import LoKiMonitorWorkflow


def loki_monitor_workflow_test() -> None:
    # Test building the sciline pipeline.
    workflow = LoKiMonitorWorkflow()
    assert isinstance(workflow.pipeline, sl.pipeline)
