# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""BIFROST workflows."""

from typing import Any

import sciline

from ess.reduce import time_of_flight
from ess.spectroscopy.indirect.conversion import providers as conversion_providers
from ess.spectroscopy.indirect.kf import providers as kf_providers
from ess.spectroscopy.indirect.ki import providers as ki_providers
from ess.spectroscopy.indirect.normalization import providers as normalisation_providers
from ess.spectroscopy.indirect.time_of_flight import TofWorkflow
from ess.spectroscopy.types import (
    DataGroupedByRotation,
    NeXusDetectorName,
    NeXusMonitorName,
    PulsePeriod,
    SampleRun,
)

from .cutting import providers as cutting_providers
from .detector import merge_triplets
from .detector import providers as detector_providers
from .io import mcstas, nexus
from .types import (
    FrameMonitor0,
    FrameMonitor1,
    FrameMonitor2,
    FrameMonitor3,
)


def simulation_default_parameters() -> dict[type, Any]:
    """Default parameters for BifrostSimulationWorkflow."""
    tof_params = time_of_flight.default_parameters()
    return {
        NeXusMonitorName[FrameMonitor0]: '007_frame_0',
        NeXusMonitorName[FrameMonitor1]: '090_frame_1',
        NeXusMonitorName[FrameMonitor2]: '097_frame_2',
        NeXusMonitorName[FrameMonitor3]: '110_frame_3',
        PulsePeriod: tof_params[PulsePeriod],
    }


_SIMULATION_PROVIDERS = (
    *nexus.providers,
    *conversion_providers,
    *detector_providers,
    *mcstas.providers,
    *cutting_providers,
    *ki_providers,
    *kf_providers,
    *normalisation_providers,
)


def BifrostSimulationWorkflow(
    detector_names: list[NeXusDetectorName],
    tof_lut_provider: time_of_flight.TofLutProvider = time_of_flight.TofLutProvider.FILE,  # noqa: E501
) -> sciline.Pipeline:
    """Data reduction workflow for simulated BIFROST data.

    Parameters
    ----------
    detector_names:
        Names of ``NXdetector`` groups in the input NeXus file.
    tof_lut_provider:
        Specifies how the time-of-flight lookup table is provided:
        - FILE: Read from a file.
        - TOF: Computed from chopper settings using the 'tof' package.
        - MCSTAS: From McStas simulation (not implemented yet).

    Returns
    -------
    :
        A pipeline for reducing simulated BIFROST data.
    """
    workflow = TofWorkflow(
        run_types=(SampleRun,),
        monitor_types=(FrameMonitor0, FrameMonitor1, FrameMonitor2, FrameMonitor3),
        tof_lut_provider=tof_lut_provider,
    )
    for provider in _SIMULATION_PROVIDERS:
        workflow.insert(provider)
    for key, val in simulation_default_parameters().items():
        workflow[key] = val

    workflow[DataGroupedByRotation[SampleRun]] = (
        workflow[DataGroupedByRotation[SampleRun]]
        .map(_make_detector_name_mapping(detector_names))
        .reduce(func=merge_triplets)
    )

    return workflow


def _make_detector_name_mapping(detector_names: list[NeXusDetectorName]) -> Any:
    # Use Pandas if possible to label the index.
    try:
        import pandas

        return pandas.DataFrame({NeXusDetectorName: detector_names}).rename_axis(
            index='triplet'
        )
    except ModuleNotFoundError:
        return {NeXusDetectorName: detector_names}
