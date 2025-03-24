# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from typing import Any

import sciline

from ess.reduce import time_of_flight
from ess.spectroscopy.indirect.conversion import providers as conversion_providers
from ess.spectroscopy.indirect.kf import providers as kf_providers
from ess.spectroscopy.indirect.ki import providers as ki_providers
from ess.spectroscopy.indirect.normalization import providers as normalisation_providers
from ess.spectroscopy.types import (
    DataGroupedByRotation,
    NeXusDetectorName,
    NeXusMonitorName,
    PulsePeriod,
    SampleRun,
)

from .detector import merge_triplets
from .detector import providers as detector_providers
from .io import mcstas, nexus
from .slicing import providers as slicing_providers
from .types import (
    FrameMonitor0,
    FrameMonitor1,
    FrameMonitor2,
    FrameMonitor3,
)


def default_parameters() -> dict[type, Any]:
    tof_params = time_of_flight.default_parameters()
    return {
        NeXusMonitorName[FrameMonitor0]: '007_frame_0',
        NeXusMonitorName[FrameMonitor1]: '090_frame_1',
        NeXusMonitorName[FrameMonitor2]: '097_frame_2',
        NeXusMonitorName[FrameMonitor3]: '110_frame_3',
        PulsePeriod: tof_params[PulsePeriod],
    }


_SIMULATION_PROVIDERS = (
    *conversion_providers,
    *detector_providers,
    *mcstas.providers,
    *slicing_providers,
    *ki_providers,
    *kf_providers,
    *normalisation_providers,
)


def BifrostSimulationWorkflow(
    detector_names: list[NeXusDetectorName],
) -> sciline.Pipeline:
    """Data reduction workflow for simulated BIFROST data."""
    workflow = nexus.LoadNeXusWorkflow()
    for provider in _SIMULATION_PROVIDERS:
        workflow.insert(provider)
    for key, val in default_parameters().items():
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
