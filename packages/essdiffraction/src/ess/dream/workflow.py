# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import itertools

import sciline
import scipp as sc
import scippnexus as snx

from ess.powder import providers as powder_providers
from ess.powder.types import (
    AccumulatedProtonCharge,
    Position,
    SampleRun,
    VanadiumRun,
)

from .io.cif import CIFAuthors, prepare_reduced_dspacing_cif
from .io.geant4 import LoadGeant4Workflow

_dream_providers = (prepare_reduced_dspacing_cif,)


def default_parameters() -> dict:
    # Quantities not available in the simulated data
    sample_position = sc.vector([0.0, 0.0, 0.0], unit="mm")
    source_position = sc.vector([-3.478, 0.0, -76550], unit="mm")
    charge = sc.scalar(1.0, unit="µAh")
    return {
        Position[snx.NXsample, SampleRun]: sample_position,
        Position[snx.NXsample, VanadiumRun]: sample_position,
        Position[snx.NXsource, SampleRun]: source_position,
        Position[snx.NXsource, VanadiumRun]: source_position,
        AccumulatedProtonCharge[SampleRun]: charge,
        AccumulatedProtonCharge[VanadiumRun]: charge,
        CIFAuthors: CIFAuthors([]),
    }


def DreamGeant4Workflow() -> sciline.Pipeline:
    """
    Workflow with default parameters for the Dream Geant4 simulation.
    """
    wf = LoadGeant4Workflow()
    for provider in itertools.chain(powder_providers, _dream_providers):
        wf.insert(provider)
    for key, value in default_parameters().items():
        wf[key] = value
    return wf


__all__ = ['DreamGeant4Workflow', 'default_parameters']
