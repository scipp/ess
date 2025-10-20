# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Contains the providers for the orca workflow.
"""

import sciline as sl
import scipp as sc

from .. import imaging
from .types import (
    CorrectedDetector,
    DarkBackgroundRun,
    NormalizedDetector,
    OpenBeamRun,
    ProtonCharge,
    RunType,
    SampleRun,
)


class ExposureTime(sl.Scope[RunType, sc.DataArray], sc.DataArray):
    """Exposure time for a run."""


def normalize_by_proton_charge_orca(
    data: CorrectedDetector[RunType],
    proton_charge: ProtonCharge[RunType],
    exposure_time: ExposureTime[RunType],
) -> NormalizedDetector[RunType]:
    """
    Normalize detector data by the proton charge (for dark and open beam runs).

    Parameters
    ----------
    data:
        Corrected detector data to be normalized.
    proton_charge:
        Proton charge data for normalization.
    """

    return NormalizedDetector[RunType](data.sum('time') / proton_charge.sum('time'))


def normalize_by_proton_charge_orca_sample(
    data: CorrectedDetector[SampleRun],
    proton_charge: ProtonCharge[SampleRun],
    exposure_time: ExposureTime[SampleRun],
) -> NormalizedDetector[SampleRun]:
    """
    Normalize sample run detector data by the proton charge.
    The handling of the SampleRun is different from the other runs:
    The sample may have a time dimension representing multiple frames.
    In this case, we need to sum the proton charge for each frame separately.

    Parameters
    ----------
    data:
        Corrected detector data to be normalized.
    proton_charge:
        Proton charge data for normalization.
    """

    # How do we do this?
    # The data can have a time dimension. We want to sum the proton charge inside each
    # time bin (defined by the duration of each frame). However, the time dimension of
    # the data recorded at the detector is not the same time as the proton charge
    # (which is when the protons hit the target). We need to shift the proton charge
    # time to account for the time it takes for neutrons to travel from the target
    # to the detector. Does this mean we can't do the normalization without computing
    # time of flight?

    time = data.coords['time']
    last_exposure_time = time[-1] - time[-2]
    time_bins = sc.concat([time, time[-1] + last_exposure_time], dim='time')
    rebinned_proton_charge = proton_charge.hist(time=time_bins)

    # Here we assume the proton charge only has a single time dimension.
    return NormalizedDetector[SampleRun](
        data / rebinned_proton_charge.drop_coords('time')
    )


# providers = (
#     normalize_by_proton_charge_orca,
#     normalize_by_proton_charge_orca_sample,
# )


def OrcaNormalizedImagesWorkflow(**kwargs) -> sl.Pipeline:
    """
    Workflow with default parameters for TBL.
    """

    wf = sl.Pipeline(
        (*imaging.normalization.providers, *imaging.masking.providers),
        constraints={RunType: [SampleRun, OpenBeamRun, DarkBackgroundRun]},
    )
    wf.insert(normalize_by_proton_charge_orca)
    wf.insert(normalize_by_proton_charge_orca_sample)
    return wf
