# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Contains the providers for the orca workflow.
"""

import sciline as sl
import scipp as sc

from ess.reduce.nexus import GenericNeXusWorkflow

from .. import imaging
from ..imaging.types import (
    AllRuns,
    CorrectedDetector,
    DarkBackgroundRun,
    FluxNormalizedDetector,
    OpenBeamRun,
    ProtonCharge,
    RunType,
    SampleRun,
)


def normalize_by_proton_charge_orca(
    data: CorrectedDetector[RunType], proton_charge: ProtonCharge[RunType]
) -> FluxNormalizedDetector[RunType]:
    """
    Normalize detector data by the proton charge.
    We find the time stamps for the data, which mark the start of an exposure,
    find the corresponding proton charge for each time stamp, and divide the data by
    the proton charge.

    Parameters
    ----------
    data:
        Corrected detector data to be normalized.
    proton_charge:
        Proton charge data for normalization.
    """

    # A note on timings:
    # We want to sum the proton charge inside each time bin (defined by the duration of
    # each frame). However, the time dimension of the data recorded at the detector is
    # not the same time as the proton charge (which is when the protons hit the
    # target). We need to shift the proton charge time to account for the time it takes
    # for neutrons to travel from the target to the detector. Does this mean we cannot
    # do the normalization without computing time of flight?

    proton_charge_lookup = sc.lookup(proton_charge, 'time', mode='previous')

    return FluxNormalizedDetector[RunType](
        data / proton_charge_lookup[data.coords['time']]
    )


orca_providers = (normalize_by_proton_charge_orca,)


def OrcaNormalizedImagesWorkflow(**kwargs) -> sl.Pipeline:
    """
    Workflow with default parameters for ORCA image normalization.
    """

    wf = GenericNeXusWorkflow(
        run_types=[SampleRun, OpenBeamRun, DarkBackgroundRun, AllRuns],
        monitor_types=[],
        **kwargs,
    )

    for provider in (
        *imaging.normalization.providers,
        *imaging.masking.providers,
        *orca_providers,
    ):
        wf.insert(provider)
    return wf
