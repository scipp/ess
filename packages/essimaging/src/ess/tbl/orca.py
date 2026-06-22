# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Contains the providers for the orca workflow.
"""

import sciline as sl
import scipp as sc

from ess.reduce.nexus import GenericNeXusWorkflow
from ess.reduce.nexus.types import NeXusDetectorName, NeXusName

from .. import imaging
from ..imaging.types import (
    CorrectedDetector,
    DarkBackgroundRun,
    ExposureTime,
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
    #
    # TODO: using the 'nearest' mode for now. Because of the above, we probably need to
    # know the start and end of the exposure, get a min and max wavelength for that
    # exposure from a wavelength lookup table, and then trace backwards to the source
    # to find the corresponding time range for the proton charge.
    proton_charge_lookup = sc.lookup(proton_charge, 'time', mode='nearest')

    return FluxNormalizedDetector[RunType](
        data / proton_charge_lookup[data.coords['time']]
    )


orca_providers = (normalize_by_proton_charge_orca,)


def default_parameters() -> dict:
    """Return the default NeXus names and detector name for the ORCA workflow."""
    return {
        NeXusDetectorName: 'orca_detector',
        NeXusName[ExposureTime]: '/entry/instrument/orca_detector/camera_exposure',
    }


def OrcaNormalizedImagesWorkflow(**kwargs) -> sl.Pipeline:
    """
    Workflow with default parameters for ORCA image normalization.
    """

    wf = GenericNeXusWorkflow(
        run_types=[SampleRun, OpenBeamRun, DarkBackgroundRun],
        monitor_types=[],
        **kwargs,
    )

    for provider in (
        *imaging.normalization.providers,
        *imaging.masking.providers,
        *orca_providers,
    ):
        wf.insert(provider)
    for key, param in default_parameters().items():
        wf[key] = param
    return wf
