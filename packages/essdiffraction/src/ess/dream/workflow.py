# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import itertools

import sciline
import scipp as sc

from ess.powder import providers as powder_providers
from ess.powder.types import (
    AccumulatedProtonCharge,
    NeXusSample,
    NeXusSource,
    SampleRun,
    VanadiumRun,
)

from .io.cif import CIFAuthors, prepare_reduced_dspacing_cif
from .io.geant4 import LoadGeant4Workflow

_dream_providers = (prepare_reduced_dspacing_cif,)


def default_parameters() -> dict:
    # Quantities not available in the simulated data
    sample = sc.DataGroup(position=sc.vector([0.0, 0.0, 0.0], unit="mm"))
    source = sc.DataGroup(position=sc.vector([-3.478, 0.0, -76550], unit="mm"))
    charge = sc.scalar(1.0, unit="µAh")
    return {
        NeXusSample[SampleRun]: sample,
        NeXusSample[VanadiumRun]: sample,
        NeXusSource[SampleRun]: source,
        NeXusSource[VanadiumRun]: source,
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
