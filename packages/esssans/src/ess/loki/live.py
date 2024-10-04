# This file is used by beamlime to create a workflow for the Loki instrument.
# The callable `live_workflow` is registered as the entry point for the workflow.
from pathlib import Path
from typing import NewType

import sciline
import scipp as sc
import scippnexus as snx

from ess import loki
from ess.reduce import streaming
from ess.reduce.nexus import types as nexus_types
from ess.reduce.nexus.json_nexus import JSONGroup
from ess.sans.types import (
    Filename,
    Incident,
    MonitorType,
    NeXusMonitorName,
    RunType,
    SampleRun,
    Transmission,
    WavelengthBins,
    WavelengthMonitor,
)


class MonitorHistogram(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
): ...


def _hist_monitor_wavelength(
    wavelength_bin: WavelengthBins, monitor: WavelengthMonitor[RunType, MonitorType]
) -> MonitorHistogram[RunType, MonitorType]:
    return monitor.hist(wavelength=wavelength_bin)


JSONEventData = NewType('JSONEventData', dict[str, JSONGroup])


def load_json_incident_monitor_data(
    name: NeXusMonitorName[Incident],
    nxevent_data: JSONEventData,
) -> nexus_types.NeXusMonitorData[SampleRun, Incident]:
    json = nxevent_data[name]
    group = snx.Group(json, definitions=snx.base_definitions())
    return nexus_types.NeXusMonitorData[SampleRun, Incident](group[()])


def load_json_transmission_monitor_data(
    name: NeXusMonitorName[Transmission],
    nxevent_data: JSONEventData,
) -> nexus_types.NeXusMonitorData[SampleRun, Incident]:
    json = nxevent_data[name]
    group = snx.Group(json, definitions=snx.base_definitions())
    return nexus_types.NeXusMonitorData[SampleRun, Incident](group[()])


class LoKiMonitorWorkflow:
    """LoKi Monitor wavelength histogram workflow for live data reduction."""

    def __init__(self, nexus_filename: Path) -> None:
        self._workflow = self._build_pipeline(nexus_filename=nexus_filename)
        self._streamed = streaming.StreamProcessor(
            base_workflow=self._workflow,
            dynamic_keys=(JSONEventData,),
            target_keys=(
                MonitorHistogram[SampleRun, Incident],
                MonitorHistogram[SampleRun, Transmission],
            ),
            accumulators=(),
        )

    def _build_pipeline(self, nexus_filename: Path) -> sciline.Pipeline:
        """Build a workflow pipeline for live data reduction.

        Returns
        -------
        :
            A pipeline for live data reduction.
            The initial pipeline will be missing the input data.
            It should be set before calling the pipeline.

        """
        workflow = loki.LokiAtLarmorWorkflow()
        workflow.insert(_hist_monitor_wavelength)
        workflow.insert(load_json_incident_monitor_data)
        workflow.insert(load_json_transmission_monitor_data)
        workflow[Filename[SampleRun]] = nexus_filename
        workflow[WavelengthBins] = sc.linspace(
            "wavelength", 1.0, 13.0, 50 + 1, unit='angstrom'
        )
        return workflow

    def __call__(
        self, nxevent_data: dict[str, JSONGroup], nxlog: dict[str, JSONGroup]
    ) -> dict[str, sc.DataArray]:
        """

        Returns
        -------
        :
            Plottable Outputs:

            - MonitorHistogram[SampleRun, Incident]
            - MonitorHistogram[SampleRun, Transmission]

        """
        # I think we will be getting the full path, but the workflow only needs the
        # name of the monitor or detector group.
        nxevent_data = {
            key.lstrip('/').split('/')[2]: value for key, value in nxevent_data.items()
        }
        required_keys = {
            self._workflow.compute(NeXusMonitorName[Incident]),
            self._workflow.compute(NeXusMonitorName[Transmission]),
        }
        if not required_keys.issubset(nxevent_data):
            raise ValueError(f"Expected {required_keys}, got {set(nxevent_data)}")
        results = self._streamed.add_chunk({JSONEventData: nxevent_data})
        return {
            'Incident Monitor': results[MonitorHistogram[SampleRun, Incident]],
            'Transmission Monitor': results[MonitorHistogram[SampleRun, Transmission]],
        }
