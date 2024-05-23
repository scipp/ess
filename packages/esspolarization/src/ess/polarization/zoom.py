# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Optional

import ess.isissans as isis
import mantid.api as _mantid_api
import sciline as sl
import scipp as sc
from ess.isissans.data import LoadedFileContents
from ess.isissans.mantidio import DataWorkspace, Period
from ess.sans.types import (
    Filename,
    Incident,
    MonitorType,
    NeXusMonitorName,
    RawMonitor,
    RunType,
    SampleRun,
    Transmission,
    TransmissionRun,
    UncertaintyBroadcastMode,
)
from mantid import simpleapi as _mantid_simpleapi

# In this case the "sample" is the analyzer cell, of which we want to measure
# the transmission fraction.
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


def _get_time(dg: sc.DataGroup) -> sc.Variable:
    start = sc.datetime(dg['run_start'].value)
    end = sc.datetime(dg['run_end'].value)
    delta = end - start
    return start + delta // 2


def _get_time_dependent_monitor(
    runs: list[sc.DataGroup], monitor_spectrum: int
) -> sc.DataArray:
    monitors = []
    for run in runs:
        # Note we index with a scipp.Variable, i.e., by the spectrum number used at ISIS
        monitor = run['data']['spectrum', sc.index(monitor_spectrum)].copy()
        monitor.coords['datetime'] = _get_time(run)
        monitors.append(monitor)
    monitors = sc.concat(monitors, 'time')
    datetime = monitors.coords['datetime']
    monitors.coords['time'] = datetime - datetime.min()
    del monitors.coords['spectrum']
    del monitors.coords['detector_id']
    monitors.variances = None
    return monitors


def get_time_dependent_incident(
    runs: sl.Series[
        Filename[sample_run_type], isis.data.LoadedFileContents[sample_run_type]
    ],
) -> RawMonitor[sample_run_type, Incident]:
    """Extract incident monitor from ZOOM direct-beam run"""
    return RawMonitor[sample_run_type, Incident](
        _get_time_dependent_monitor(list(runs.values()), monitor_spectrum=3)
    )


def get_time_dependent_transmission(
    runs: sl.Series[
        Filename[sample_run_type], isis.data.LoadedFileContents[sample_run_type]
    ],
) -> RawMonitor[sample_run_type, Transmission]:
    """Extract transmission monitor from ZOOM direct-beam run"""
    return RawMonitor[sample_run_type, Transmission](
        _get_time_dependent_monitor(list(runs.values()), monitor_spectrum=5)
    )


def get_monitor_data(
    dg: LoadedFileContents[RunType], nexus_name: NeXusMonitorName[MonitorType]
) -> RawMonitor[RunType, MonitorType]:
    # See https://github.com/scipp/sciline/issues/52 why copy needed
    mon = dg['monitors'][nexus_name]['data'].copy()
    # TODO This is a hack to work around broadcasting issues of variances when
    # computing the transmission fraction.
    return RawMonitor[RunType, MonitorType](sc.values(mon))


def ZoomTransmissionFractionWorkflow() -> sl.Pipeline:
    """
    Workflow computing time-dependent SANS transmission fraction from ZOOM data.

    The time-dependence is obtained by using a series of runs. This should be set as
    a parameter series:

    .. code-block:: python

        workflow = ZoomTransmissionFractionWorkflow()
        workflow.set_param_series(Filename[TransmissionRun[SampleRun]], cell_runs)

    Note that in this case the "sample" (of which the transmission is to be computed)
    is the He3 analyzer cell.
    """
    workflow = isis.zoom.ZoomWorkflow()
    workflow.insert(get_monitor_data)
    workflow.insert(load_histogrammed_run)
    workflow.insert(get_time_dependent_incident)
    workflow.insert(get_time_dependent_transmission)
    workflow[NeXusMonitorName[Incident]] = 'monitor3'
    workflow[NeXusMonitorName[Transmission]] = 'monitor5'
    workflow[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound

    return workflow
