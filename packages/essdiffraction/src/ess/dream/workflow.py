# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import sciline
import scipp as sc
from ess.powder import providers as powder_providers
from ess.powder.types import (
    AccumulatedProtonCharge,
    RawSample,
    RawSource,
    SampleRun,
    VanadiumRun,
)

from .io.geant4 import providers as geant4_providers


def default_parameters() -> dict:
    # Quantities not available in the simulated data
    sample = sc.DataGroup(position=sc.vector([0.0, 0.0, 0.0], unit="mm"))
    source = sc.DataGroup(position=sc.vector([-3.478, 0.0, -76550], unit="mm"))
    charge = sc.scalar(1.0, unit="µAh")
    return {
        RawSample[SampleRun]: sample,
        RawSample[VanadiumRun]: sample,
        RawSource[SampleRun]: source,
        RawSource[VanadiumRun]: source,
        AccumulatedProtonCharge[SampleRun]: charge,
        AccumulatedProtonCharge[VanadiumRun]: charge,
    }


def DreamGeant4Workflow() -> sciline.Pipeline:
    """
    Workflow with default parameters for the Powgen SNS instrument.
    """
    return sciline.Pipeline(
        providers=powder_providers + geant4_providers, params=default_parameters()
    )


__all__ = ['DreamGeant4Workflow', 'default_parameters']
