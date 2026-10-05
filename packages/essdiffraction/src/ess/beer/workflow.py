# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import sciline as sl
import scipp as sc
import scippnexus as snx
from ess.powder import binning as powder_binning
from ess.powder import providers as powder_providers
from ess.powder.calibration import detector_two_theta
from ess.powder.conversion import powder_coordinate_transformation_graph
from ess.powder.correction import RunNormalization, insert_run_normalization
from ess.powder.types import (
    BunkerMonitor,
    CalibrationData,
    CaveMonitor,
    EmptyCanRun,
    SampleRun,
    TwoThetaBins,
    VanadiumRun,
)

from ess.reduce.nexus import GenericNeXusWorkflow
from ess.reduce.nexus.types import DetectorBankSizes, NeXusName
from ess.reduce.unwrap import (
    GenericUnwrapWorkflow,
    PulsePeriod,
    SourceBounds,
)
from ess.reduce.unwrap import lut as unwrap_lut
from ess.reduce.unwrap.types import LookupTableRelativeErrorThreshold

from .clustering import cluster_events_by_streak
from .conversions import (
    automatic_coordinate_transformation_graph,
    compute_wavelength_in_each_cluster,
    known_peaks_coordinate_transformation_graph,
    wavelength_detector,
)
from .mcstas import mcstas_providers
from .types import DetectorBank, PulseLength

default_parameters = {
    CalibrationData: None,
    TwoThetaBins: None,
    PulseLength: sc.scalar(0.003, unit='s'),
    PulsePeriod: 1.0 / sc.scalar(14.0, unit='Hz'),
    SourceBounds: SourceBounds(
        time=(sc.scalar(0.0, unit='ms'), sc.scalar(5.0, unit='ms')),
        wavelength=(
            sc.scalar(0.001, unit='angstrom'),
            sc.scalar(15.0, unit='angstrom'),
        ),
    ),
    DetectorBankSizes: {
        'south_detector': {'y': 200, 'x': 500},
        'north_detector': {'y': 200, 'x': 500},
    },
    DetectorBank: DetectorBank.both,
}


def _insert_dspacing_range_detection(workflow: sl.Pipeline) -> None:
    """Add automatic d-spacing range detection to a BEER workflow."""
    # Bin edges need chopper frames regardless of how event wavelengths are computed.
    for provider in (
        unwrap_lut.get_active_choppers,
        unwrap_lut.close_non_synced_disk_choppers,
        unwrap_lut.guess_pulse_stride_from_choppers,
        unwrap_lut.compute_frame_sequence,
        *powder_binning.providers,
        detector_two_theta,
    ):
        workflow.insert(provider)


def _mcstas_beer_modulation_workflow(graph_provider, *providers) -> sl.Pipeline:
    workflow = GenericNeXusWorkflow(run_types=[SampleRun], monitor_types=[])
    for provider in (
        *mcstas_providers,
        graph_provider,
        *providers,
    ):
        workflow.insert(provider)
    for key, value in default_parameters.items():
        workflow[key] = value
    _insert_dspacing_range_detection(workflow)
    return workflow


def BeerModMcStasWorkflow():
    """Process modulation-mode McStas data without known peak positions."""
    return _mcstas_beer_modulation_workflow(
        automatic_coordinate_transformation_graph,
        cluster_events_by_streak,
        compute_wavelength_in_each_cluster,
    )


def BeerModMcStasWorkflowKnownPeaks():
    """Process modulation-mode McStas data using known peak positions."""
    return _mcstas_beer_modulation_workflow(
        known_peaks_coordinate_transformation_graph, wavelength_detector
    )


def BeerMcStasWorkflowPulseShaping():
    """Workflow to process BEER pulse-shaping McStas files using analytical
    frame unwrapping."""
    wf = GenericUnwrapWorkflow(
        run_types=[SampleRun], monitor_types=[], wavelength_from='analytical'
    )
    for provider in (*mcstas_providers, powder_coordinate_transformation_graph):
        wf.insert(provider)
    for key, value in default_parameters.items():
        wf[key] = value
    _insert_dspacing_range_detection(wf)
    wf[NeXusName[snx.NXdetector]] = 'detector'
    wf[LookupTableRelativeErrorThreshold] = {'detector': float('inf')}
    return wf


def BeerPowderWorkflow(
    *, run_norm: RunNormalization = RunNormalization.monitor_integrated, **kwargs
) -> sl.Pipeline:
    """
    Beer powder workflow with default parameters.

    Parameters
    ----------
    run_norm:
        Select how to normalize each run (sample, vanadium, etc.).
    kwargs:
        Additional keyword arguments are forwarded to the base
        :func:`GenericUnwrapWorkflow`.

    Returns
    -------
    :
        A workflow object for BEER.
    """
    wf = GenericUnwrapWorkflow(
        run_types=[SampleRun, VanadiumRun, EmptyCanRun],
        monitor_types=[BunkerMonitor, CaveMonitor],
        **kwargs,
    )
    wf[NeXusName[CaveMonitor]] = "monitor_cave"

    for provider in powder_providers:
        wf.insert(provider)
    _insert_dspacing_range_detection(wf)

    insert_run_normalization(wf, run_norm)
    for key, value in default_parameters.items():
        wf[key] = value
    return wf


def BeerPowderWorkflowAnalytical(
    *, run_norm: RunNormalization = RunNormalization.monitor_integrated, **kwargs
) -> sl.Pipeline:
    """
    Beer powder workflow using analytical lookup-table frame unwrapping.

    Parameters
    ----------
    run_norm:
        Select how to normalize each run (sample, vanadium, etc.).
    kwargs:
        Additional keyword arguments are forwarded to the base
        :func:`GenericUnwrapWorkflow`.

    Returns
    -------
    :
        A workflow object for BEER.
    """
    wf = BeerPowderWorkflow(
        run_norm=run_norm,
        wavelength_from='analytical',
        **kwargs,
    )
    wf[NeXusName[snx.NXdetector]] = 'detector'
    wf[LookupTableRelativeErrorThreshold] = {
        'detector': float('inf'),
        'monitor_bunker': float('inf'),
        'monitor_cave': float('inf'),
    }
    return wf


def BeerPowderMcStasWorkflow(**kwargs) -> sl.Pipeline:
    """Create the BEER analytical powder workflow with McStas loaders inserted."""
    wf = BeerPowderWorkflowAnalytical(**kwargs)
    for provider in mcstas_providers:
        wf.insert(provider)

    return wf
