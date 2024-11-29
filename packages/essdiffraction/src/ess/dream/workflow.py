# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from __future__ import annotations

import itertools

import sciline
import scipp as sc
import scippnexus as snx

from ess.powder import providers as powder_providers
from ess.powder import with_pixel_mask_filenames
from ess.powder.correction import (
    RunNormalization,
    insert_run_normalization,
)
from ess.powder.types import (
    AccumulatedProtonCharge,
    CaveMonitorPosition,  # Should this be a DREAM-only parameter?
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

from .io.cif import CIFAuthors, prepare_reduced_tof_cif
from .io.geant4 import LoadGeant4Workflow
from .parameters import typical_outputs

_dream_providers = (prepare_reduced_tof_cif,)

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
        CaveMonitorPosition: sc.vector([0.0, 0.0, -4220.0], unit='mm'),
    }


def DreamGeant4Workflow(*, run_norm: RunNormalization) -> sciline.Pipeline:
    """
    Workflow with default parameters for the Dream Geant4 simulation.
    """
    wf = LoadGeant4Workflow()
    for provider in itertools.chain(powder_providers, _dream_providers):
        wf.insert(provider)
    insert_run_normalization(wf, run_norm)
    for key, value in default_parameters().items():
        wf[key] = value
    wf.typical_outputs = typical_outputs
    return wf


@register_workflow
def DreamGeant4MonitorHistogramWorkflow() -> sciline.Pipeline:
    """
    Workflow with default parameters for the Dream Geant4 simulation, using a
    histogrammed monitor for the normalization.
    """
    return DreamGeant4Workflow(run_norm=RunNormalization.monitor_histogram)


@register_workflow
def DreamGeant4MonitorIntegratedWorkflow() -> sciline.Pipeline:
    """
    Workflow with default parameters for the Dream Geant4 simulation, using
    integrated counts of the monitor for the normalization.
    """
    return DreamGeant4Workflow(run_norm=RunNormalization.monitor_integrated)


@register_workflow
def DreamGeant4ProtonChargeWorkflow() -> sciline.Pipeline:
    """
    Workflow with default parameters for the Dream Geant4 simulation, using
    proton charge for the normalization.
    """
    return DreamGeant4Workflow(run_norm=RunNormalization.proton_charge)


__all__ = [
    'DreamGeant4MonitorHistogramWorkflow',
    'DreamGeant4MonitorIntegratedWorkflow',
    'DreamGeant4ProtonChargeWorkflow',
    'DreamGeant4Workflow',
    'default_parameters',
]
