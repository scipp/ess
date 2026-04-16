# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import itertools

import sciline as sl
import scipp as sc

from ess.powder import providers as powder_providers
from ess.powder.correction import RunNormalization, insert_run_normalization
from ess.powder.types import (
    BunkerMonitor,
    CaveMonitor,
    EmptyCanRun,
    RunType,
    SampleRun,
    VanadiumRun,
)
from ess.reduce.nexus.types import DetectorBankSizes, NeXusName
from ess.reduce.unwrap import GenericUnwrapWorkflow

from .clustering import providers as clustering_providers
from .conversions import convert_from_known_peaks_providers, convert_pulse_shaping
from .conversions import providers as conversion_providers
from .io import mcstas_modulation_period_from_mode, mcstas_providers
from .types import (
    PulseLength,
)

default_parameters = {
    PulseLength: sc.scalar(0.003, unit='s'),
}


def BeerModMcStasWorkflow():
    """Workflow to process BEER (modulation regime) McStas files without a list
    of estimated peak positions."""
    return sl.Pipeline(
        (
            *mcstas_providers,
            mcstas_modulation_period_from_mode,
            *clustering_providers,
            *conversion_providers,
        ),
        params=default_parameters,
        constraints={RunType: (SampleRun,)},
    )


def BeerModMcStasWorkflowKnownPeaks():
    """Workflow to process BEER (modulation regime) McStas files using a list
    of estimated peak positions."""
    return sl.Pipeline(
        (
            *mcstas_providers,
            mcstas_modulation_period_from_mode,
            *convert_from_known_peaks_providers,
        ),
        params=default_parameters,
        constraints={RunType: (SampleRun,)},
    )


def BeerMcStasWorkflowPulseShaping():
    """Workflow to process BEER (pulse shaping modes) McStas files"""
    return sl.Pipeline(
        (*mcstas_providers, *convert_pulse_shaping),
        params=default_parameters,
        constraints={RunType: (SampleRun,)},
    )


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
    wf[DetectorBankSizes] = {
        'south_detector': {'y': 200, 'x': 500},
        'north_detector': {'y': 200, 'x': 500},
    }
    wf[NeXusName[CaveMonitor]] = "monitor_cave"

    for provider in itertools.chain(powder_providers, convert_pulse_shaping):
        wf.insert(provider)

    insert_run_normalization(wf, run_norm)
    for key, value in default_parameters.items():
        wf[key] = value
    return wf


def BeerPowderMcStasWorkflow(**kwargs) -> sl.Pipeline:
    wf = BeerPowderWorkflow(**kwargs)
    for provider in mcstas_providers:
        wf.insert(provider)

    return wf
