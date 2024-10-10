# This file is used by beamlime to create a workflow for the Loki instrument.
# The callable `live_workflow` is registered as the entry point for the workflow.
"""
Live data reduction workflows for LoKI.
"""

from pathlib import Path
from typing import NewType

import sciline
import scipp as sc

import ess.loki.data  # noqa: F401
from ess import loki
from ess.reduce import streaming
from ess.reduce.live import LiveWorkflow
from ess.sans import with_pixel_mask_filenames
from ess.sans.types import (
    BackgroundRun,
    BackgroundSubtractedIofQ,
    BackgroundSubtractedIofQxy,
    BeamCenter,
    CorrectForGravity,
    Denominator,
    DetectorData,
    DirectBeamFilename,
    EmptyBeamRun,
    Filename,
    Incident,
    IofQ,
    IofQxy,
    Numerator,
    QBins,
    QxBins,
    QyBins,
    ReducedQ,
    ReducedQxy,
    ReturnEvents,
    RunType,
    SampleRun,
    Transmission,
    TransmissionFraction,
    TransmissionRun,
    UncertaintyBroadcastMode,
    WavelengthBins,
    WavelengthMonitor,
)

RawDetectorView = NewType('RawDetectorView', sc.DataArray)


def _raw_detector_view(data: DetectorData[SampleRun]) -> RawDetectorView:
    """Very simple raw detector view for initial testing."""
    # Instead of histogramming concrete x and y (which leads to artifacts), another
    # quick option is to slice/sum some dimensions. But it will not give true positions:
    # Option 1:
    # return data.hist().sum(('straw', 'layer'))
    # Option 2:
    # return data['layer', 0].hist().sum('straw')
    da = data.hist()
    da.coords['x'] = da.coords['position'].fields.x.copy()
    da.coords['y'] = da.coords['position'].fields.y.copy()
    return da.hist(y=50, x=100)


class GatheredMonitors(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup): ...


def _gather_monitors(
    incident: WavelengthMonitor[RunType, Incident],
    transmission: WavelengthMonitor[RunType, Transmission],
) -> GatheredMonitors[RunType]:
    """
    Helper to allow for plotting multiple monitors in a single plot.

    Might be better to handle this via a nested output spec in LiveWorkflow?
    """
    return GatheredMonitors[RunType](
        sc.DataGroup(
            {'Incident Monitor': incident, 'Transmission Monitor': transmission}
        )
    )


def _configured_Larmor_workflow() -> sciline.Pipeline:
    wf = loki.LokiAtLarmorWorkflow()
    wf = with_pixel_mask_filenames(wf, masks=loki.data.loki_tutorial_mask_filenames())
    wf[CorrectForGravity] = True
    wf[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    wf[ReturnEvents] = False

    wf[WavelengthBins] = sc.linspace('wavelength', 1.0, 13.0, 201, unit='angstrom')
    wf[QBins] = sc.linspace(dim='Q', start=0.01, stop=0.3, num=101, unit='1/angstrom')
    wf[QxBins] = sc.linspace(dim='Qx', start=-0.3, stop=0.3, num=61, unit='1/angstrom')
    wf[QyBins] = sc.linspace(dim='Qy', start=-0.3, stop=0.3, num=61, unit='1/angstrom')

    wf[Filename[EmptyBeamRun]] = loki.data.loki_tutorial_run_60392()
    return wf


def _configured_Larmor_AgBeh_workflow() -> sciline.Pipeline:
    wf = _configured_Larmor_workflow()

    # AgBeh
    wf[BeamCenter] = sc.vector(value=[-0.0295995, -0.02203635, 0.0], unit='m')
    wf[DirectBeamFilename] = loki.data.loki_tutorial_direct_beam_all_pixels()
    wf[Filename[EmptyBeamRun]] = loki.data.loki_tutorial_run_60392()
    wf[Filename[TransmissionRun[BackgroundRun]]] = loki.data.loki_tutorial_run_60392()
    wf[Filename[BackgroundRun]] = loki.data.loki_tutorial_background_run_60393()
    wf[Filename[TransmissionRun[SampleRun]]] = (
        loki.data.loki_tutorial_agbeh_transmission_run()
    )
    wf[Filename[SampleRun]] = loki.data.loki_tutorial_agbeh_sample_run()
    return wf


class AccumulatorFactories:
    """Helper to create accumulator factories with different preprocessors."""

    def __init__(self, accum: type[streaming.Accumulator], **kwargs) -> None:
        self._accum = accum
        self._kwargs = kwargs

    def with_hist(self) -> streaming.Accumulator:
        return self._accum(**self._kwargs, preprocess=streaming.maybe_hist)

    def with_wavelength_hist(self, wav: WavelengthBins) -> streaming.Accumulator:
        return self._accum(**self._kwargs, preprocess=lambda x: x.hist(wavelength=wav))


def make_monitor_workflow(
    nexus_filename: Path, workflow: sciline.Pipeline
) -> LiveWorkflow:
    """Loki monitor wavelength histogram workflow for live data reduction."""
    # By adding accumulators we obtain automatic histogramming of our outputs.
    factories = AccumulatorFactories(accum=streaming.RollingAccumulator, window=1)
    factory = factories.with_wavelength_hist
    return LiveWorkflow.from_workflow(
        workflow=workflow,
        accumulators={
            WavelengthMonitor[SampleRun, Incident]: factory,
            WavelengthMonitor[SampleRun, Transmission]: factory,
        },
        outputs={
            'Incident Monitor': WavelengthMonitor[SampleRun, Incident],
            'Transmission Monitor': WavelengthMonitor[SampleRun, Transmission],
        },
        run_type=SampleRun,
        nexus_filename=nexus_filename,
    )


def make_transmission_run_workflow(
    nexus_filename: Path, workflow: sciline.Pipeline
) -> LiveWorkflow:
    """Loki transmission run workflow for live data reduction."""
    workflow.insert(_gather_monitors)
    factories = AccumulatorFactories(accum=streaming.EternalAccumulator)
    factory = factories.with_wavelength_hist
    return LiveWorkflow.from_workflow(
        workflow=workflow,
        accumulators={
            WavelengthMonitor[TransmissionRun[SampleRun], Incident]: factory,
            WavelengthMonitor[TransmissionRun[SampleRun], Transmission]: factory,
        },
        outputs={
            'Monitors (cumulative)': GatheredMonitors[TransmissionRun[SampleRun]],
            'Transmission Fraction': TransmissionFraction[SampleRun],
        },
        run_type=TransmissionRun[SampleRun],
        nexus_filename=nexus_filename,
    )


def make_loki_workflow(
    nexus_filename: Path, workflow: sciline.Pipeline
) -> LiveWorkflow:
    """Loki workflow for live data reduction."""
    workflow.insert(_raw_detector_view)
    outputs = {'Raw Detector': RawDetectorView}
    try:
        workflow.compute(Filename[BackgroundRun])
    except sciline.UnsatisfiedRequirement:
        iofq_keys = (IofQ[SampleRun], IofQxy[SampleRun])
    else:
        iofq_keys = (BackgroundSubtractedIofQ, BackgroundSubtractedIofQxy)
    outputs.update(dict(zip(('I(Q)', '$I(Q_x, Q_y)$'), iofq_keys, strict=True)))
    factories = AccumulatorFactories(accum=streaming.RollingAccumulator, window=20)

    return LiveWorkflow.from_workflow(
        workflow=workflow,
        accumulators={
            ReducedQ[SampleRun, Numerator]: factories.with_hist,
            ReducedQ[SampleRun, Denominator]: factories.with_hist,
            ReducedQxy[SampleRun, Numerator]: factories.with_hist,
            ReducedQxy[SampleRun, Denominator]: factories.with_hist,
            RawDetectorView: factories.with_hist,
        },
        outputs=outputs,
        run_type=SampleRun,
        nexus_filename=nexus_filename,
    )


def LokiMonitorTestWorkflow(nexus_filename: Path) -> LiveWorkflow:
    """Fully preconfigured monitor workflow for, testing with Beamlime."""
    return make_monitor_workflow(nexus_filename, _configured_Larmor_workflow())


def LokiTransmissionRunTestWorkflow(nexus_filename: Path) -> LiveWorkflow:
    """Fully preconfigured transmission run workflow, for testing with Beamlime."""
    return make_transmission_run_workflow(nexus_filename, _configured_Larmor_workflow())


def LokiAtLarmorAgBehTestWorkflow(nexus_filename: Path) -> LiveWorkflow:
    """Fully preconfigured I(Q) workflow for AgBeh, for testing with Beamlime."""
    return make_loki_workflow(nexus_filename, _configured_Larmor_AgBeh_workflow())
