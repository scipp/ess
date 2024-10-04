# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import pytest
import scipp as sc
import scippnexus as snx

from ess.loki import data
from ess.loki.live import LoKiMonitorWorkflow
from ess.reduce.nexus.json_generator import event_data_generator
from ess.reduce.nexus.json_nexus import JSONGroup


def test_can_create_loki_monitor_workflow() -> None:
    filename = data.loki_tutorial_sample_run_60250()
    _ = LoKiMonitorWorkflow(filename)


def test_workflow_processes_monitor_event_data_chunks() -> None:
    filename = data.loki_tutorial_sample_run_60250()
    mon1 = snx.load(filename, root='entry/instrument/monitor_1/monitor_1_events')
    mon2 = snx.load(filename, root='entry/instrument/monitor_2/monitor_2_events')
    generator1 = event_data_generator(mon1)
    generator2 = event_data_generator(mon2)
    wf = LoKiMonitorWorkflow(filename)

    group1 = JSONGroup(next(generator1))
    group2 = JSONGroup(next(generator2))
    result = sc.DataGroup(wf({'monitor_1': group1, 'monitor_2': group2}, {}))
    assert list(result) == ['Incident Monitor', 'Transmission Monitor']

    # If we pass the same data again, we should get the same result, there is no
    # accumulation in this workflow.
    same = sc.DataGroup(wf({'monitor_1': group1, 'monitor_2': group2}, {}))
    assert sc.identical(same, result)

    group1 = JSONGroup(next(generator1))
    group2 = JSONGroup(next(generator2))
    different = sc.DataGroup(wf({'monitor_1': group1, 'monitor_2': group2}, {}))
    assert not sc.identical(different, result)


def test_workflow_raises_if_event_data_missing() -> None:
    filename = data.loki_tutorial_sample_run_60250()
    mon1 = snx.load(filename, root='entry/instrument/monitor_1/monitor_1_events')
    mon2 = snx.load(filename, root='entry/instrument/monitor_2/monitor_2_events')
    generator1 = event_data_generator(mon1)
    generator2 = event_data_generator(mon2)
    wf = LoKiMonitorWorkflow(filename)

    group1 = JSONGroup(next(generator1))
    group2 = JSONGroup(next(generator2))
    result = sc.DataGroup(wf({'monitor_1': group1, 'monitor_2': group2}, {}))
    assert list(result) == ['Incident Monitor', 'Transmission Monitor']

    group1 = JSONGroup(next(generator1))
    group2 = JSONGroup(next(generator2))
    with pytest.raises(ValueError, match="Expected"):
        wf({'monitor_2': group2}, {})
