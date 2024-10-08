# This file is used by beamlime to create a workflow for the Loki instrument.
# The callable `live_workflow` is registered as the entry point for the workflow.
from pathlib import Path
from typing import NewType, TypeVar

import sciline
import scipp as sc
import scippnexus as snx

import ess.loki.data  # noqa: F401
from ess import loki
from ess.reduce import streaming
from ess.reduce.nexus import types as nexus_types
from ess.reduce.nexus.json_nexus import JSONGroup
from ess.sans import with_pixel_mask_filenames
from ess.sans.types import (
    BackgroundRun,
    BeamCenter,
    CorrectForGravity,
    Denominator,
    DirectBeamFilename,
    EmptyBeamRun,
    Filename,
    Incident,
    IofQ,
    IofQxy,
    MonitorType,
    Numerator,
    QBins,
    QxBins,
    QyBins,
    ReducedQ,
    ReturnEvents,
    RunType,
    SampleRun,
    Transmission,
    TransmissionRun,
    UncertaintyBroadcastMode,
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


def load_json_event_data(
    name: nexus_types.NeXusName[nexus_types.Component],
    nxevent_data: JSONEventData,
) -> nexus_types.NeXusData[nexus_types.Component, SampleRun]:
    json = nxevent_data[name]
    group = snx.Group(json, definitions=snx.base_definitions())
    return nexus_types.NeXusData[nexus_types.Component, SampleRun](group[()])


T = TypeVar('T', bound='LiveWorkflow')


class LiveWorkflow:
    def __init__(
        self,
        *,
        streamed: streaming.StreamProcessor,
        outputs: dict[str, sciline.typing.Key],
    ) -> None:
        self._streamed = streamed
        self._outputs = outputs

    @classmethod
    def from_workflow(
        cls: type[T],
        *,
        workflow: sciline.Pipeline,
        accumulators: dict[sciline.typing.Key, streaming.Accumulator],
        outputs: dict[str, sciline.typing.Key],
        nexus_filename: Path,
    ) -> T:
        workflow = workflow.copy()
        # Do we need to generalize this function and Filename key for other run types?
        # For now they are both hard-coded to "SampleRun".
        workflow.insert(load_json_event_data)
        workflow[Filename[SampleRun]] = nexus_filename
        streamed = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(JSONEventData,),
            target_keys=outputs.values(),
            accumulators=accumulators,
        )
        return cls(streamed=streamed, outputs=outputs)

    def __call__(
        self, nxevent_data: dict[str, JSONGroup], nxlog: dict[str, JSONGroup]
    ) -> dict[str, sc.DataArray]:
        # Beamlime passes full path, but the workflow only needs the name of the monitor
        # or detector group.
        nxevent_data = {
            key.lstrip('/').split('/')[2]: value for key, value in nxevent_data.items()
        }
        results = self._streamed.add_chunk({JSONEventData: nxevent_data})
        return {name: results[key] for name, key in self._outputs.items()}


def LokiMonitorWorkflow(nexus_filename: Path) -> LiveWorkflow:
    """Loki monitor wavelength histogram workflow for live data reduction."""
    workflow = loki.LokiAtLarmorWorkflow()
    workflow.insert(_hist_monitor_wavelength)
    workflow[WavelengthBins] = sc.linspace(
        "wavelength", 1.0, 13.0, 50 + 1, unit='angstrom'
    )
    return LiveWorkflow.from_workflow(
        workflow=workflow,
        accumulators={},
        outputs={
            'Incident Monitor': MonitorHistogram[SampleRun, Incident],
            'Transmission Monitor': MonitorHistogram[SampleRun, Transmission],
        },
        nexus_filename=nexus_filename,
    )


def LokiAtLarmorAgBehWorkflow(nexus_filename: Path) -> LiveWorkflow:
    """Loki workflow for live data reduction."""
    workflow = loki.LokiAtLarmorWorkflow()
    workflow = with_pixel_mask_filenames(
        workflow, masks=loki.data.loki_tutorial_mask_filenames()
    )
    workflow.insert(_hist_monitor_wavelength)

    workflow[CorrectForGravity] = True
    workflow[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    workflow[ReturnEvents] = False

    workflow[WavelengthBins] = sc.linspace(
        "wavelength", 1.0, 13.0, 50 + 1, unit='angstrom'
    )
    workflow[QBins] = sc.linspace(
        dim='Q', start=0.01, stop=0.3, num=101, unit='1/angstrom'
    )
    workflow[QxBins] = sc.linspace(
        dim='Qx', start=-0.3, stop=0.3, num=61, unit='1/angstrom'
    )
    workflow[QyBins] = sc.linspace(
        dim='Qy', start=-0.3, stop=0.3, num=61, unit='1/angstrom'
    )

    # AgBeh
    workflow[BeamCenter] = sc.vector(value=[-0.0295995, -0.02203635, 0.0], unit='m')
    workflow[DirectBeamFilename] = loki.data.loki_tutorial_direct_beam_all_pixels()
    workflow[Filename[EmptyBeamRun]] = loki.data.loki_tutorial_run_60392()
    workflow[Filename[BackgroundRun]] = loki.data.loki_tutorial_background_run_60393()

    # TODO The transmission monitor may actually be unused (but gets computed
    # anyway!) if # a separate transmission run is provided. We also need a way to
    # be able to set the # transmission monitor data for the transmission run if
    # such a run is not available.
    workflow[Filename[TransmissionRun[SampleRun]]] = (
        loki.data.loki_tutorial_agbeh_transmission_run()
    )
    workflow[Filename[TransmissionRun[BackgroundRun]]] = (
        loki.data.loki_tutorial_run_60392()
    )
    return LiveWorkflow.from_workflow(
        workflow=workflow,
        accumulators={
            ReducedQ[SampleRun, Numerator]: streaming.RollingAccumulator(window=20),
            ReducedQ[SampleRun, Denominator]: streaming.RollingAccumulator(window=20),
        },
        outputs={
            'Incident Monitor': MonitorHistogram[SampleRun, Incident],
            'Transmission Monitor': MonitorHistogram[SampleRun, Transmission],
            'I(Q)': IofQ[SampleRun],
            'I(Q_x, Q_y)': IofQxy[SampleRun],
        },
        nexus_filename=nexus_filename,
    )
