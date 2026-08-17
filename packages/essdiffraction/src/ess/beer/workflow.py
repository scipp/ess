# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import sciline as sl
import scipp as sc
import scippnexus as snx
from ess.powder import providers as powder_providers
from ess.powder.conversion import powder_coordinate_transformation_graph
from ess.powder.correction import RunNormalization, insert_run_normalization
from ess.powder.types import (
    BunkerMonitor,
    CalibrationData,
    CaveMonitor,
    EmptyCanRun,
    SampleRun,
    VanadiumRun,
)

from ess.reduce.nexus import GenericNeXusWorkflow
from ess.reduce.nexus.types import DetectorBankSizes, NeXusName
from ess.reduce.unwrap import GenericUnwrapWorkflow
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
    PulseLength: sc.scalar(0.003, unit='s'),
    DetectorBankSizes: {
        'south_detector': {'y': 200, 'x': 500},
        'north_detector': {'y': 200, 'x': 500},
    },
    DetectorBank: DetectorBank.both,
}


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
