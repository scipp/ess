# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from typing import Any

import sciline
import scippnexus as snx

from ess.reduce import time_of_flight
from ess.spectroscopy.types import (
    CalibratedDetector,
    DetectorPositionOffset,
    NeXusComponent,
    NeXusMonitorName,
    NeXusTransformation,
    PulsePeriod,
    RunType,
)

from .io import mcstas, nexus
from .types import (
    FrameMonitor0,
    FrameMonitor1,
    FrameMonitor2,
    FrameMonitor3,
)


def get_calibrated_detector_bifrost(
    detector: NeXusComponent[snx.NXdetector, RunType],
    *,
    transform: NeXusTransformation[snx.NXdetector, RunType],
    offset: DetectorPositionOffset[RunType],
) -> CalibratedDetector[RunType]:
    """Extract the data array corresponding to a detector's signal field.

    The returned data array includes coords and masks pertaining directly to the
    signal values array, but not additional information about the detector.
    The data array is reshaped to the logical detector shape.

    This function is specific to BIFROST and differs from the generic
    :func:`ess.reduce.nexus.workflow.get_calibrated_detector` in that it does not
    fold the detectors into logical dimensions because the files already contain
    the detectors in the correct shape.

    Parameters
    ----------
    detector:
        Loaded NeXus detector.
    transform:
        Transformation that determines the detector position.
    offset:
        Offset to add to the detector position.

    Returns
    -------
    :
        Detector data.
    """

    from ess.reduce.nexus.types import DetectorBankSizes
    from ess.reduce.nexus.workflow import get_calibrated_detector

    da = get_calibrated_detector(
        detector=detector,
        transform=transform,
        offset=offset,
        # The detectors are folded in the file, no need to do that here.
        bank_sizes=DetectorBankSizes({}),
    )
    da = da.rename(dim_0='tube', dim_1='length')
    return CalibratedDetector[RunType](da)


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
    get_calibrated_detector_bifrost,
    *mcstas.providers,
)


def BifrostSimulationWorkflow() -> sciline.Pipeline:
    """Data reduction workflow for simulated BIFROST data."""
    workflow = nexus.LoadNeXusWorkflow()
    for provider in _SIMULATION_PROVIDERS:
        workflow.insert(provider)
    for key, val in default_parameters().items():
        workflow[key] = val
    return workflow
