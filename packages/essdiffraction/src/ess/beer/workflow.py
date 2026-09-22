# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import warnings

import sciline as sl
import scipp as sc
import scippnexus as snx
from ess.powder import providers as powder_providers
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
from ess.reduce.unwrap import GenericUnwrapWorkflow, WavelengthLutMode
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
    DetectorBankSizes: {
        'south_detector': {'y': 200, 'x': 500},
        'north_detector': {'y': 200, 'x': 500},
    },
    DetectorBank: DetectorBank.both,
}


def _beer_modulation_workflow(
    graph_provider,
    *providers,
    run_norm: RunNormalization = RunNormalization.monitor_integrated,
) -> sl.Pipeline:
    workflow = GenericNeXusWorkflow(
        run_types=[SampleRun],
        monitor_types=[BunkerMonitor, CaveMonitor],
    )
    for provider in (*powder_providers, graph_provider, *providers):
        workflow.insert(provider)
    insert_run_normalization(workflow, run_norm)
    for key, value in default_parameters.items():
        workflow[key] = value
    return workflow


def BeerModulationAutoMcStasWorkflow(
    run_norm: RunNormalization = RunNormalization.monitor_integrated,
) -> sl.Pipeline:
    """Process modulation-mode McStas data without known peak positions."""
    workflow = _beer_modulation_workflow(
        automatic_coordinate_transformation_graph,
        cluster_events_by_streak,
        compute_wavelength_in_each_cluster,
        run_norm=run_norm,
    )
    for provider in mcstas_providers:
        workflow.insert(provider)
    return workflow


def BeerModulationKnownPeaksMcStasWorkflow(
    run_norm: RunNormalization = RunNormalization.monitor_integrated,
) -> sl.Pipeline:
    """Process modulation-mode McStas data using known peak positions."""
    workflow = _beer_modulation_workflow(
        known_peaks_coordinate_transformation_graph,
        wavelength_detector,
        run_norm=run_norm,
    )
    for provider in mcstas_providers:
        workflow.insert(provider)
    return workflow


def BeerPowderWorkflow(
    *,
    run_norm: RunNormalization = RunNormalization.monitor_integrated,
    wavelength_from: WavelengthLutMode = 'analytical',
) -> sl.Pipeline:
    """
    Beer powder workflow with default parameters.

    Parameters
    ----------
    run_norm:
        Select how to normalize each run (sample, vanadium, etc.).
    wavelength_from:
        Mode for creating the wavelength lookup table. Defaults to analytical.

    Returns
    -------
    :
        A workflow object for BEER.
    """
    wf = GenericUnwrapWorkflow(
        run_types=[SampleRun, VanadiumRun, EmptyCanRun],
        monitor_types=[BunkerMonitor, CaveMonitor],
        wavelength_from=wavelength_from,
    )
    wf[NeXusName[CaveMonitor]] = 'monitor_cave'
    wf[NeXusName[snx.NXdetector]] = 'detector'
    wf[LookupTableRelativeErrorThreshold] = {
        'detector': float('inf'),
        'monitor_bunker': float('inf'),
        'monitor_cave': float('inf'),
    }

    for provider in powder_providers:
        wf.insert(provider)

    insert_run_normalization(wf, run_norm)
    for key, value in default_parameters.items():
        wf[key] = value
    return wf


def BeerPowderMcStasWorkflow(
    *, run_norm: RunNormalization = RunNormalization.monitor_integrated
) -> sl.Pipeline:
    """Create the BEER analytical powder workflow with McStas loaders inserted."""
    wf = BeerPowderWorkflow(run_norm=run_norm)
    for provider in mcstas_providers:
        wf.insert(provider)

    return wf


def BeerModMcStasWorkflow() -> sl.Pipeline:
    """Deprecated: use :func:`BeerModulationAutoMcStasWorkflow`."""
    warnings.warn(
        'BeerModMcStasWorkflow is deprecated; use '
        'BeerModulationAutoMcStasWorkflow instead.',
        DeprecationWarning,
        stacklevel=2,
    )
    return BeerModulationAutoMcStasWorkflow()


def BeerModMcStasWorkflowKnownPeaks() -> sl.Pipeline:
    """Deprecated: use :func:`BeerModulationKnownPeaksMcStasWorkflow`."""
    warnings.warn(
        'BeerModMcStasWorkflowKnownPeaks is deprecated; use '
        'BeerModulationKnownPeaksMcStasWorkflow instead.',
        DeprecationWarning,
        stacklevel=2,
    )
    return BeerModulationKnownPeaksMcStasWorkflow()


def BeerMcStasWorkflowPulseShaping() -> sl.Pipeline:
    """Deprecated: use :func:`BeerPowderMcStasWorkflow`."""
    warnings.warn(
        'BeerMcStasWorkflowPulseShaping is deprecated; use '
        'BeerPowderMcStasWorkflow instead.',
        DeprecationWarning,
        stacklevel=2,
    )
    return BeerPowderMcStasWorkflow()


def BeerPowderWorkflowAnalytical(
    *, run_norm: RunNormalization = RunNormalization.monitor_integrated, **kwargs
) -> sl.Pipeline:
    """Deprecated: use :func:`BeerPowderWorkflow` with analytical wavelength lookup."""
    warnings.warn(
        'BeerPowderWorkflowAnalytical is deprecated; use '
        "BeerPowderWorkflow(wavelength_from='analytical') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return BeerPowderWorkflow(run_norm=run_norm, wavelength_from='analytical', **kwargs)
