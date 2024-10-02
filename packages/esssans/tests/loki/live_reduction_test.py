# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import sciline as sl

from ess.loki.live import LoKiMonitorWorkflow


def test_loki_monitor_workflow() -> None:
    # Test building the sciline pipeline.
    workflow = LoKiMonitorWorkflow()
    assert isinstance(workflow.pipeline, sl.Pipeline)
