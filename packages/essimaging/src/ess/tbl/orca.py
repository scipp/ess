# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Contains the providers for the orca workflow.
"""

import sciline as sl
import scipp as sc

from ess.reduce.nexus import GenericNeXusWorkflow, load_from_path
from ess.reduce.nexus.types import (
    NeXusDetectorName,
    NeXusFileSpec,
    NeXusLocationSpec,
    NeXusName,
)

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


def load_exposure_time(
    file: NeXusFileSpec[RunType], path: NeXusName[ExposureTime[RunType]]
) -> ExposureTime[RunType]:
    # Note that putting '/value' at the end of the 'path' in the default_parameters
    # yields different results as it can return a Variable instead of a DataArray,
    # depending on the contents of the NeXus file.
    return ExposureTime[RunType](
        load_from_path(NeXusLocationSpec(filename=file.value, component_name=path))[
            "value"
        ].squeeze()
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


providers = (load_exposure_time, normalize_by_proton_charge_orca)


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
        *providers,
    ):
        wf.insert(provider)
    for key, param in default_parameters().items():
        wf[key] = param
    return wf
