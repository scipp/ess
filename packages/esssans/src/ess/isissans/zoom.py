# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from collections.abc import Sequence

import sciline
import scipp as sc
from ess.sans import SansWorkflow
from ess.sans.io import read_xml_detector_masking
from ess.sans.parameters import typical_outputs
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
from scippnexus import NXsource

from ess.reduce.nexus.types import Position
from ess.reduce.workflow import register_workflow

from .general import MonitorSpectrumNumber, default_parameters, get_monitor_data
from .io import LoadedFileContents, load_tutorial_direct_beam, load_tutorial_run
from .mantidio import providers as mantid_providers

# In this case the "sample" is the analyzer cell, of which we want to measure
# the transmission fraction.
sample_run_type = RunType


def set_mantid_log_level(level: int = 3):
    try:
        from mantid import ConfigService

        cfg = ConfigService.Instance()
        cfg.setLogLevel(level)  # Silence verbose load via Mantid
    except ImportError:
        pass


@register_workflow
def ZoomWorkflow() -> sciline.Pipeline:
    """Create Zoom workflow with default parameters."""
    from . import providers as isis_providers

    set_mantid_log_level()

    # Note that the actual NeXus loading in this workflow will not be used for the
    # ISIS files, the providers inserted below will replace those steps.
    workflow = SansWorkflow()
    for provider in isis_providers + mantid_providers:
        workflow.insert(provider)
    for key, param in default_parameters().items():
        workflow[key] = param
    workflow.insert(read_xml_detector_masking)
    workflow.typical_outputs = typical_outputs
    return workflow


@register_workflow
def ZoomTutorialWorkflow() -> sciline.Pipeline:
    """
    Create Zoom tutorial workflow.

    Equivalent to :func:`ZoomWorkflow`, but with loaders for tutorial data instead
    of Mantid-based loaders.
    """
    workflow = ZoomWorkflow()
    workflow.insert(load_tutorial_run)
    workflow.insert(load_tutorial_direct_beam)
    return workflow


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
    spectrum_number: MonitorSpectrumNumber[MonitorType],
) -> NeXusComponent[MonitorType, RunType]:
    """
    Same as :py:func:`ess.isissans.get_monitor_data` but dropping variances.
    """
    monitor = get_monitor_data(
        dg, nexus_name=nexus_name, spectrum_number=spectrum_number
    )
    monitor['data'] = sc.values(monitor['data'])
    return NeXusComponent[MonitorType, RunType](monitor)


def get_monitor_data_from_transmission_run(
    dg: LoadedFileContents[TransmissionRun[RunType]],
    spectrum_number: MonitorSpectrumNumber[MonitorType],
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


def ZoomTransmissionFractionWorkflow(runs: Sequence[str]) -> sciline.Pipeline:
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
    workflow = ZoomWorkflow()
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
