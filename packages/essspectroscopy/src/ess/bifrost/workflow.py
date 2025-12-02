# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""BIFROST workflows."""

from typing import Any

import sciline
import scipp as sc
from scippnexus import NXdetector

from ess.spectroscopy.indirect.conversion import providers as conversion_providers
from ess.spectroscopy.indirect.kf import providers as kf_providers
from ess.spectroscopy.indirect.ki import providers as ki_providers
from ess.spectroscopy.indirect.normalization import providers as normalisation_providers
from ess.spectroscopy.indirect.time_of_flight import TofWorkflow
from ess.spectroscopy.types import (
    EmptyDetector,
    FrameMonitor0,
    FrameMonitor1,
    FrameMonitor2,
    FrameMonitor3,
    NeXusData,
    NeXusDetectorName,
    NeXusMonitorName,
    PulsePeriod,
    RawDetector,
    SampleRun,
)

from .cutting import providers as cutting_providers
from .detector import merge_triplets
from .detector import providers as detector_providers
from .io import mcstas, nexus


def default_parameters() -> dict[type, Any]:
    """Default parameters for BifrostWorkflow."""
    return {
        NeXusMonitorName[FrameMonitor1]: '090_frame_1',
        NeXusMonitorName[FrameMonitor2]: '097_frame_2',
        NeXusMonitorName[FrameMonitor3]: '110_frame_3',
        PulsePeriod: 1.0 / sc.scalar(14.0, unit="Hz"),
    }


def simulation_default_parameters() -> dict[type, Any]:
    """Default parameters for BifrostSimulationWorkflow."""
    return {
        NeXusMonitorName[FrameMonitor1]: '090_frame_1',
        NeXusMonitorName[FrameMonitor2]: '097_frame_2',
        NeXusMonitorName[FrameMonitor3]: '110_frame_3',
        PulsePeriod: 1.0 / sc.scalar(14.0, unit="Hz"),
    }


_PROVIDERS = (
    *nexus.providers,
    *conversion_providers,
    *detector_providers,
    *cutting_providers,
    *ki_providers,
    *kf_providers,
    *normalisation_providers,
)

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

    workflow[RawDetector[SampleRun]] = (
        workflow[RawDetector[SampleRun]]
        .map(_make_detector_name_mapping(detector_names))
        .reduce(func=merge_triplets)
    )

    return workflow


def BifrostWorkflow(
    detector_names: list[NeXusDetectorName],
) -> sciline.Pipeline:
    """Data reduction workflow for BIFROST."""
    workflow = TofWorkflow(
        run_types=(SampleRun,),
        monitor_types=(FrameMonitor0, FrameMonitor1, FrameMonitor2, FrameMonitor3),
    )
    for provider in _PROVIDERS:
        workflow.insert(provider)
    for key, val in default_parameters().items():
        workflow[key] = val

    workflow[EmptyDetector[SampleRun]] = (
        workflow[EmptyDetector[SampleRun]]
        .map(_make_detector_name_mapping(detector_names))
        .reduce(func=merge_triplets)
    )

    workflow[NeXusData[NXdetector, SampleRun]] = (
        workflow[NeXusData[NXdetector, SampleRun]]
        .map(_make_detector_name_mapping(detector_names))
        .reduce(func=concat_event_lists)
    )

    return workflow


# TODO remove or move
def concat_event_lists(
    *data: sc.DataArray,
) -> sc.DataArray:
    """Concatenate binned event lists into a single data array in 'event_time_zero'.

    Note that the output will likely have repeated values for 'event_time_zero'.
    E.g., if input 1 has times `[0, 1, 2]` and input 2 has times `[0, 2, 3]`,
    the output will have times `[0, 1, 2, 0, 2, 3]`.
    Note that this sawtooth pattern will disappear again after grouping into pixels.
    Preserving it will likely actually lead to more efficient memory
    access patterns when grouping.

    Parameters
    ----------
    data:
        Data arrays to concatenate.
        Must be binned in 'event_time_zero'.

    Returns
    -------
    :
        Concatenated data array.
    """
    return sc.concat(data, dim="event_time_zero")


def _make_detector_name_mapping(detector_names: list[NeXusDetectorName]) -> Any:
    # Use Pandas if possible to label the index.
    try:
        import pandas

        return pandas.DataFrame({NeXusDetectorName: detector_names}).rename_axis(
            index='triplet'
        )
    except ModuleNotFoundError:
        return {NeXusDetectorName: detector_names}
