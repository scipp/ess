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


def load_proton_charge(
    file: NeXusFileSpec[RunType], path: NeXusName[ProtonCharge[RunType]]
) -> ProtonCharge[RunType]:
    # Note that putting '/value' at the end of the 'path' in the default_parameters
    # yields different results as it can return a Variable instead of a DataArray,
    # depending on the contents of the NeXus file.
    return ProtonCharge[RunType](
        load_from_path(NeXusLocationSpec(filename=file.value, component_name=path))[
            "value"
        ]
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


def _compute_proton_charge_per_exposure(
    data: sc.DataArray, proton_charge: sc.DataArray, exposure_time: sc.DataArray
) -> sc.DataArray:
    # A note on timings:
    # We want to sum the proton charge inside each time bin (defined by the duration of
    # each frame). However, the time dimension of the data recorded at the detector is
    # not the same time as the proton charge (which is when the protons hit the
    # target). We need to shift the proton charge time to account for the time it takes
    # for neutrons to travel from the target to the detector. Does this mean we cannot
    # do the normalization without computing time of flight?

    t = data.coords['time']
    exp = exposure_time.data.to(unit=t.unit)
    # The following assumes that the different between successive frames (time stamps)
    # is larger than the exposure time. We need to check that this is indeed the case.
    if (t[1:] - t[:-1]).min() < exp:
        raise ValueError(
            "normalize_by_proton_charge_orca: Exposure time is larger than the "
            "smallest time between successive frames."
        )

    bins = sc.sort(sc.concat([t, t + exp], dim=t.dim), t.dim)
    # We select every second bin, as the odd bins lie between the end of the exposure
    # of one frame and the start of the next frame.
    return proton_charge.hist(time=bins).data[::2]


def normalize_by_proton_charge_orca(
    data: CorrectedDetector[RunType],
    proton_charge: ProtonCharge[RunType],
    exposure_time: ExposureTime[RunType],
) -> FluxNormalizedDetector[RunType]:
    """
    Normalize detector data by the proton charge (dark and open beam runs).
    We find the time stamps for the data, which mark the start of an exposure.
    We sum the proton charge within the intervals timestamp to timestamp +
    exposure_time.
    We then sum the data along the time dimension, and divide by the sum of the proton
    charge accumulated during each exposure.

    Parameters
    ----------
    data:
        Corrected detector data to be normalized.
    proton_charge:
        Proton charge data for normalization.
    exposure_time:
        Exposure time for each image in the data.
    """

    charge_per_frame = _compute_proton_charge_per_exposure(
        data, proton_charge, exposure_time
    )

    return FluxNormalizedDetector[RunType](
        data.sum('time') / charge_per_frame.sum('time')
    )


def normalize_by_proton_charge_orca_sample(
    data: CorrectedDetector[SampleRun],
    proton_charge: ProtonCharge[SampleRun],
    exposure_time: ExposureTime[SampleRun],
) -> FluxNormalizedDetector[SampleRun]:
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

    charge_per_frame = _compute_proton_charge_per_exposure(
        data, proton_charge, exposure_time
    )

    # Here we preserve the time dimension of the sample data and the proton charge.
    return FluxNormalizedDetector[SampleRun](data / charge_per_frame)


providers = (
    load_exposure_time,
    load_proton_charge,
    normalize_by_proton_charge_orca,
    normalize_by_proton_charge_orca_sample,
)


def default_parameters() -> dict:
    return {
        NeXusDetectorName: 'orca_detector',
        NeXusName[ProtonCharge]: '/entry/neutron_prod_info/pulse_charge',
        NeXusName[ExposureTime]: '/entry/instrument/orca_detector/camera_exposure',
    }


def OrcaNormalizedImagesWorkflow(**kwargs) -> sl.Pipeline:
    """
    Workflow with default parameters for TBL.
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
