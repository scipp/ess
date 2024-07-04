# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import Generic, Sequence

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
    filename: Filename[sample_run_type], period: Period
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


def _get_time_dependent_monitor(*monitors: sc.DataArray) -> sc.DataArray:
    monitors = sc.concat(monitors, 'time')
    datetime = monitors.coords['datetime']
    monitors.coords['time'] = datetime - datetime.min()
    del monitors.coords['spectrum']
    del monitors.coords['detector_id']
    return monitors


@dataclass
class MonitorSpectrumNumber(Generic[MonitorType]):
    value: int


def get_monitor_data(
    dg: LoadedFileContents[RunType], nexus_name: NeXusMonitorName[MonitorType]
) -> RawMonitor[RunType, MonitorType]:
    """
    Same as :py:func:`ess.isissans.get_monitor_data` but dropping variances.

    Dropping variances is a workaround required since ESSsans does not handle
    variance broadcasting when combining monitors. In our case some of the monitors
    are time-dependent, so this is required for now.
    """
    # See https://github.com/scipp/sciline/issues/52 why copy needed
    mon = dg['monitors'][nexus_name]['data'].copy()
    return RawMonitor[RunType, MonitorType](sc.values(mon))


def get_monitor_data_from_transmission_run(
    dg: LoadedFileContents[TransmissionRun[RunType]],
    spectrum_number: MonitorSpectrumNumber[MonitorType],
) -> RawMonitor[TransmissionRun[RunType], MonitorType]:
    """
    Extract incident or transmission monitor from ZOOM direct-beam run

    The files in this case do not contain detector data, only monitor data. Mantid
    stores this as a Workspace2D, where each spectrum corresponds to a monitor.
    """
    # Note we index with a scipp.Variable, i.e., by the spectrum number used at ISIS
    monitor = dg['data']['spectrum', sc.index(spectrum_number.value)].copy()
    monitor.coords['datetime'] = _get_time(dg)
    return monitor


def ZoomTransmissionFractionWorkflow(runs: Sequence[str]) -> sl.Pipeline:
    """
    Workflow computing time-dependent SANS transmission fraction from ZOOM data.

    The time-dependence is obtained by using a sequence of runs.

    .. code-block:: python

        workflow = ZoomTransmissionFractionWorkflow(cell_runs)

    Note that in this case the "sample" (of which the transmission is to be computed)
    is the He3 analyzer cell.

    Parameters
    ----------
    runs:
        List of filenames of the runs to use for the transmission fraction.
    """
    workflow = isis.zoom.ZoomWorkflow()
    workflow.insert(get_monitor_data)
    workflow.insert(get_monitor_data_from_transmission_run)
    workflow.insert(load_histogrammed_run)

    mapped = workflow.map({Filename[TransmissionRun[SampleRun]]: runs})
    for mon_type in (Incident, Transmission):
        workflow[RawMonitor[TransmissionRun[SampleRun], mon_type]] = mapped[
            RawMonitor[TransmissionRun[SampleRun], mon_type]
        ].reduce(func=_get_time_dependent_monitor)

    # We are dealing with two different types of files, and monitors are identified
    # differently in each case, so there is some duplication here.
    workflow[MonitorSpectrumNumber[Incident]] = MonitorSpectrumNumber[Incident](3)
    workflow[MonitorSpectrumNumber[Transmission]] = MonitorSpectrumNumber[Transmission](
        5
    )
    workflow[NeXusMonitorName[Incident]] = NeXusMonitorName[Incident]('monitor3')
    workflow[NeXusMonitorName[Transmission]] = NeXusMonitorName[Transmission](
        'monitor5'
    )
    workflow[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound

    return workflow
