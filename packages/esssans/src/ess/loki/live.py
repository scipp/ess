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
    MonitorType,
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


class MonitorHistogram(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
): ...


def _hist_monitor_wavelength(
    wavelength_bin: WavelengthBins, monitor: WavelengthMonitor[RunType, MonitorType]
) -> MonitorHistogram[RunType, MonitorType]:
    return monitor.hist(wavelength=wavelength_bin)


RawDetectorView = NewType('RawDetectorView', sc.DataArray)


def _raw_detector_view(data: DetectorData[SampleRun]) -> RawDetectorView:
    da = data.hist()
    da.coords['x'] = da.coords['position'].fields.x.copy()
    da.coords['y'] = da.coords['position'].fields.y.copy()
    return da.hist(y=50, x=100)
    return data.hist().sum(('straw', 'layer'))
    return data['layer', 0].hist().sum('straw')


_wavelength = sc.linspace("wavelength", 1.0, 13.0, 200 + 1, unit='angstrom')


def _hist_wavelength(
    da: sc.DataArray, wavelength: sc.Variable = _wavelength
) -> sc.DataArray:
    return da.hist(wavelength=wavelength)


class GatheredMonitors(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup): ...


def _gather_monitors(
    incident: WavelengthMonitor[RunType, Incident],
    transmission: WavelengthMonitor[RunType, Transmission],
) -> GatheredMonitors[RunType]:
    return GatheredMonitors[RunType](
        sc.DataGroup(
            {'Incident Monitor': incident, 'Transmission Monitor': transmission}
        )
    )


def _configured_Larmor_AgBeh_workflow() -> sciline.Pipeline:
    wf = loki.LokiAtLarmorWorkflow()
    wf = with_pixel_mask_filenames(wf, masks=loki.data.loki_tutorial_mask_filenames())
    wf[CorrectForGravity] = True
    wf[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    wf[ReturnEvents] = False

    wf[WavelengthBins] = _wavelength
    wf[QBins] = sc.linspace(dim='Q', start=0.01, stop=0.3, num=101, unit='1/angstrom')
    wf[QxBins] = sc.linspace(dim='Qx', start=-0.3, stop=0.3, num=61, unit='1/angstrom')
    wf[QyBins] = sc.linspace(dim='Qy', start=-0.3, stop=0.3, num=61, unit='1/angstrom')

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


def LokiMonitorWorkflow(nexus_filename: Path) -> LiveWorkflow:
    """Loki monitor wavelength histogram workflow for live data reduction."""
    workflow = loki.LokiAtLarmorWorkflow()
    workflow.insert(_hist_monitor_wavelength)
    workflow[WavelengthBins] = _wavelength
    return LiveWorkflow.from_workflow(
        workflow=workflow,
        accumulators={},
        outputs={
            'Incident Monitor': MonitorHistogram[SampleRun, Incident],
            'Transmission Monitor': MonitorHistogram[SampleRun, Transmission],
        },
        run_type=SampleRun,
        nexus_filename=nexus_filename,
    )


def _eternal_wav_hist(wav: WavelengthBins) -> streaming.Accumulator:
    return streaming.EternalAccumulator(preprocess=lambda x: x.hist(wavelength=wav))


def LokiTransmissionRunWorkflow(nexus_filename: Path) -> LiveWorkflow:
    """Loki transmission run workflow for live data reduction."""
    workflow = loki.LokiAtLarmorWorkflow()
    workflow.insert(_gather_monitors)
    workflow[WavelengthBins] = _wavelength
    workflow[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    workflow[Filename[EmptyBeamRun]] = loki.data.loki_tutorial_run_60392()
    return LiveWorkflow.from_workflow(
        workflow=workflow,
        accumulators={
            WavelengthMonitor[TransmissionRun[SampleRun], Incident]: _eternal_wav_hist,
            WavelengthMonitor[
                TransmissionRun[SampleRun], Transmission
            ]: _eternal_wav_hist,
        },
        outputs={
            'Monitors (cumulative)': GatheredMonitors[TransmissionRun[SampleRun]],
            'Transmission Fraction': TransmissionFraction[SampleRun],
        },
        run_type=TransmissionRun[SampleRun],
        nexus_filename=nexus_filename,
    )


def LokiAtLarmorAgBehWorkflow(nexus_filename: Path) -> LiveWorkflow:
    """Loki workflow for live data reduction."""
    workflow = _configured_Larmor_AgBeh_workflow()
    workflow.insert(_hist_monitor_wavelength)
    workflow.insert(_raw_detector_view)
    outputs = {'Raw Detector': RawDetectorView}
    try:
        workflow.compute(Filename[BackgroundRun])
    except sciline.UnsatisfiedRequirement:
        iofq_keys = (IofQ[SampleRun], IofQxy[SampleRun])
    else:
        iofq_keys = (BackgroundSubtractedIofQ, BackgroundSubtractedIofQxy)
    outputs.update(dict(zip(('I(Q)', '$I(Q_x, Q_y)$'), iofq_keys, strict=True)))

    return LiveWorkflow.from_workflow(
        workflow=workflow,
        accumulators={
            ReducedQ[SampleRun, Numerator]: streaming.RollingAccumulator(window=20),
            ReducedQ[SampleRun, Denominator]: streaming.RollingAccumulator(window=20),
            ReducedQxy[SampleRun, Numerator]: streaming.RollingAccumulator(window=20),
            ReducedQxy[SampleRun, Denominator]: streaming.RollingAccumulator(window=20),
            RawDetectorView: streaming.RollingAccumulator(window=20),
        },
        outputs=outputs,
        run_type=SampleRun,
        nexus_filename=nexus_filename,
    )
