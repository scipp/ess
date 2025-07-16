# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""BIFROST workflows."""

from typing import Any

import sciline
import scipp as sc

from ess.spectroscopy.indirect.conversion import providers as conversion_providers
from ess.spectroscopy.indirect.kf import providers as kf_providers
from ess.spectroscopy.indirect.ki import providers as ki_providers
from ess.spectroscopy.indirect.normalization import providers as normalisation_providers
from ess.spectroscopy.indirect.time_of_flight import TofWorkflow
from ess.spectroscopy.types import (
    DataGroupedByRotation,
    FrameMonitor0,
    FrameMonitor1,
    FrameMonitor2,
    FrameMonitor3,
    NeXusDetectorName,
    NeXusMonitorName,
    PulsePeriod,
    SampleRun,
)

from .cutting import providers as cutting_providers
from .detector import merge_triplets
from .detector import providers as detector_providers
from .io import mcstas, nexus


def simulation_default_parameters() -> dict[type, Any]:
    """Default parameters for BifrostSimulationWorkflow."""
    return {
        NeXusMonitorName[FrameMonitor0]: '007_frame_0',
        NeXusMonitorName[FrameMonitor1]: '090_frame_1',
        NeXusMonitorName[FrameMonitor2]: '097_frame_2',
        NeXusMonitorName[FrameMonitor3]: '110_frame_3',
        PulsePeriod: 1.0 / sc.scalar(14.0, unit="Hz"),
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
) -> sciline.Pipeline:
    """Data reduction workflow for simulated BIFROST data.

    Parameters
    ----------
    detector_names:
        Names of ``NXdetector`` groups in the input NeXus file.

    Returns
    -------
    :
        A pipeline for reducing simulated BIFROST data.
    """
    workflow = TofWorkflow(
        run_types=(SampleRun,),
        monitor_types=(FrameMonitor0, FrameMonitor1, FrameMonitor2, FrameMonitor3),
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
