# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from typing import Any

import sciline
import scippnexus as snx

from ess.spectroscopy.types import (
    CalibratedDetector,
    DetectorPositionOffset,
    Monitor3,
    NeXusComponent,
    NeXusMonitorName,
    NeXusTransformation,
    RunType,
)

from .io import nexus


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
    return {
        NeXusMonitorName[Monitor3]: '110_frame_3',
    }


def BifrostSimulationWorkflow() -> sciline.Pipeline:
    """Data reduction workflow for simulated BIFROST data."""
    workflow = nexus.LoadNeXusWorkflow()
    workflow.insert(get_calibrated_detector_bifrost)
    for key, val in default_parameters().items():
        workflow[key] = val
    return workflow
