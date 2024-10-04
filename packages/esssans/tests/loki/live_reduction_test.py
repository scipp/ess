# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from ess.loki import data
from ess.loki.live import LoKiMonitorWorkflow


def test_can_create_loki_monitor_workflow() -> None:
    filename = data.loki_tutorial_sample_run_60250()
    _ = LoKiMonitorWorkflow(filename)
