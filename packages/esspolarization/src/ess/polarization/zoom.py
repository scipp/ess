# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Optional

import ess.isissans as isis
import mantid.api as _mantid_api
import sciline as sl
import scipp as sc
from ess.isissans.mantidio import DataWorkspace, Period
from ess.sans.types import (
    Filename,
    Incident,
    RawMonitor,
    SampleRun,
    Transmission,
    TransmissionRun,
)
from mantid import simpleapi as _mantid_simpleapi

sample_run_type = TransmissionRun[SampleRun]


def load_histogrammed_run(
    filename: Filename[sample_run_type], period: Optional[Period]
) -> DataWorkspace[sample_run_type]:
    """Load a non-event-data ISIS file"""
    loaded = _mantid_simpleapi.Load(Filename=str(filename), StoreInADS=False)
    if isinstance(loaded, _mantid_api.Workspace):
        # A single workspace
        data_ws = loaded
        if isinstance(data_ws, _mantid_api.WorkspaceGroup):
            if period is None:
                raise ValueError(
                    f'Needs {Period} to be set to know what '
                    'section of the event data to load'
                )
            data_ws = data_ws.getItem(period)
    else:
        # Separate data and monitor workspaces
        data_ws = loaded.OutputWorkspace
        if isinstance(data_ws, _mantid_api.WorkspaceGroup):
            if period is None:
                raise ValueError(
                    f'Needs {Period} to be set to know what '
                    'section of the event data to load'
                )
            data_ws = data_ws.getItem(period)
            data_ws.setMonitorWorkspace(loaded.MonitorWorkspace.getItem(period))
        else:
            data_ws.setMonitorWorkspace(loaded.MonitorWorkspace)
    return DataWorkspace[sample_run_type](data_ws)


def get_incident(
    dg: isis.data.LoadedFileContents[sample_run_type],
) -> RawMonitor[sample_run_type, Incident]:
    """Extract indcident monitor from ZOOM direct-beam run"""
    return RawMonitor[sample_run_type, Incident](dg['data']['spectrum', 2].copy())


def _get_time(dg: sc.DataGroup) -> sc.Variable:
    start = sc.datetime(dg['run_start'].value)
    end = sc.datetime(dg['run_end'].value)
    delta = end - start
    return start + delta // 2


def get_transmission(
    dg: isis.data.LoadedFileContents[sample_run_type],
) -> RawMonitor[sample_run_type, Transmission]:
    """Extract transmission monitor from ZOOM direct-beam run"""
    monitor = dg['data']['spectrum', 4].copy()
    monitor.coords['datetime'] = _get_time(dg)
    return RawMonitor[sample_run_type, Transmission](monitor)


def ZoomTransmissionFractionWorkflow() -> sl.Pipeline:
    """
    Workflow computing SANS transmission fraction from ZOOM data.
    """
    steps = ()
    workflow = sl.Pipeline(providers=steps)
    return workflow
