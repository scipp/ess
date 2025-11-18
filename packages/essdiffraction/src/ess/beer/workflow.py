# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import sciline as sl
import scipp as sc

from .clustering import providers as clustering_providers
from .conversions import convert_from_known_peaks_providers, convert_pulse_shaping
from .conversions import providers as conversion_providers
from .io import mcstas_modulation_period_from_mode, mcstas_providers
from .types import (
    PulseLength,
    RunType,
    SampleRun,
    TwoThetaLimits,
)

default_parameters = {
    PulseLength: sc.scalar(0.003, unit='s'),
    TwoThetaLimits: (
        sc.scalar(75, unit='deg').to(unit='rad', dtype='float64'),
        sc.scalar(105, unit='deg').to(unit='rad', dtype='float64'),
    ),
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
