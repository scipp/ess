# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import sciline

from ess.powder import providers as powder_providers
from ess.powder.types import DetectorData, NeXusDetectorName, RunType, TofData

from . import beamline


def default_parameters() -> dict:
    return {NeXusDetectorName: "powgen_detector"}


def dummy_compute_detector_time_of_flight(
    detector_data: DetectorData[RunType],
) -> TofData[RunType]:
    """
    Dummy function to compute time-of-flight data from detector data.
    The Powgen data already contains a `tof` coordinate.
    We also do not have the chopper information to compute a better estimate.

    Parameters
    ----------
    detector_data:
        Data from the detector.
    """
    return TofData[RunType](detector_data)


providers = (dummy_compute_detector_time_of_flight,)


def PowgenWorkflow() -> sciline.Pipeline:
    """
    Workflow with default parameters for the Powgen SNS instrument.
    """
    # The package does not depend on pooch which is needed for the tutorial
    # data. Delay import until workflow is actually used.
    from . import data

    return sciline.Pipeline(
        providers=powder_providers + beamline.providers + data.providers + providers,
        params=default_parameters(),
    )


__all__ = ['PowgenWorkflow', 'default_parameters', 'providers']
