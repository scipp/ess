# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import itertools

import sciline
import scipp as sc
import scippnexus as snx

from ess.powder import providers as powder_providers
from ess.powder import with_pixel_mask_filenames
from ess.powder.types import (
    AccumulatedProtonCharge,
    PixelMaskFilename,
    Position,
    SampleRun,
    TofMask,
    TwoThetaMask,
    VanadiumRun,
    WavelengthMask,
)
from ess.reduce.parameter import parameter_mappers
from ess.reduce.workflow import register_workflow

from .io.cif import CIFAuthors, prepare_reduced_dspacing_cif
from .io.geant4 import LoadGeant4Workflow
from .parameters import typical_outputs

_dream_providers = (prepare_reduced_dspacing_cif,)

parameter_mappers[PixelMaskFilename] = with_pixel_mask_filenames


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
        TofMask: None,
        WavelengthMask: None,
        TwoThetaMask: None,
    }


@register_workflow
def DreamGeant4Workflow() -> sciline.Pipeline:
    """
    Workflow with default parameters for the Dream Geant4 simulation.
    """
    wf = LoadGeant4Workflow()
    for provider in itertools.chain(powder_providers, _dream_providers):
        wf.insert(provider)
    for key, value in default_parameters().items():
        wf[key] = value
    wf.typical_outputs = typical_outputs
    return wf


__all__ = ['DreamGeant4Workflow', 'default_parameters']
