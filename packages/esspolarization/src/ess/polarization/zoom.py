# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from collections.abc import Sequence

import sciline as sl
import scipp as sc
from scippnexus import NXsource

import ess.isissans as isis
from ess.isissans.io import LoadedFileContents
from ess.reduce.nexus.types import Position
from ess.sans.types import (
    Filename,
    Incident,
    MonitorType,
    NeXusComponent,
    NeXusMonitorName,
    RunType,
    SampleRun,
    Transmission,
    TransmissionRun,
)

# In this case the "sample" is the analyzer cell, of which we want to measure
# the transmission fraction.
sample_run_type = RunType


def _get_time(dg: sc.DataGroup) -> sc.Variable:
    start = sc.datetime(dg['run_start'].value)
    end = sc.datetime(dg['run_end'].value)
    delta = end - start
    return start + delta // 2


def _get_time_dependent_monitor(*monitor_groups: sc.DataGroup) -> sc.DataGroup:
    monitors = [grp['data'] for grp in monitor_groups]
    monitor = sc.concat(monitors, 'time')
    positions = [grp['position'] for grp in monitor_groups]
    position = _get_unique_position(*positions)
    datetime = monitor.coords['datetime']
    monitor.coords['time'] = datetime - datetime.min()
    del monitor.coords['spectrum']
    del monitor.coords['detector_id']
    return sc.DataGroup(data=monitor, position=position)


def _get_unique_position(*positions: sc.DataArray) -> sc.DataArray:
    unique = positions[0]
    for position in positions[1:]:
        if not sc.identical(position, unique):
            raise ValueError("Monitors have different source positions")
    return unique


def get_monitor_data_no_variances(
    dg: LoadedFileContents[RunType],
    nexus_name: NeXusMonitorName[MonitorType],
    spectrum_number: isis.MonitorSpectrumNumber[MonitorType],
) -> NeXusComponent[MonitorType, RunType]:
    """
    Same as :py:func:`ess.isissans.get_monitor_data` but dropping variances.
    """
    monitor = isis.general.get_monitor_data(
        dg, nexus_name=nexus_name, spectrum_number=spectrum_number
    )
    monitor['data'] = sc.values(monitor['data'])
    return NeXusComponent[MonitorType, RunType](monitor)


def get_monitor_data_from_transmission_run(
    dg: LoadedFileContents[TransmissionRun[RunType]],
    spectrum_number: isis.MonitorSpectrumNumber[MonitorType],
) -> NeXusComponent[MonitorType, TransmissionRun[RunType]]:
    """
    Extract incident or transmission monitor from ZOOM direct-beam run

    The files in this case do not contain detector data, only monitor data. Mantid
    stores this as a Workspace2D, where each spectrum corresponds to a monitor.
    """
    # Note we index with a scipp.Variable, i.e., by the spectrum number used at ISIS
    monitor = dg['data']['spectrum', sc.index(spectrum_number.value)].copy()
    monitor.coords['datetime'] = _get_time(dg)
    return sc.DataGroup(data=monitor, position=monitor.coords['position'])


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
    workflow.insert(get_monitor_data_no_variances)
    workflow.insert(get_monitor_data_from_transmission_run)

    mapped = workflow.map({Filename[TransmissionRun[SampleRun]]: runs})
    for mon_type in (Incident, Transmission):
        workflow[NeXusComponent[mon_type, TransmissionRun[SampleRun]]] = mapped[
            NeXusComponent[mon_type, TransmissionRun[SampleRun]]
        ].reduce(func=_get_time_dependent_monitor)
        workflow[Position[NXsource, TransmissionRun[SampleRun]]] = mapped[
            Position[NXsource, TransmissionRun[SampleRun]]
        ].reduce(func=_get_unique_position)

    return workflow
