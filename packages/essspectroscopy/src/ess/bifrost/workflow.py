# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from typing import Any

import sciline
import scippnexus as snx

from ess.spectroscopy.types import (
    CalibratedDetector,
    DetectorPositionOffset,
    GravityVector,
    Monitor3,
    NeXusComponentLocationSpec,
    NeXusDetector,
    NeXusMonitorName,
    NeXusSource,
    RunType,
    SamplePosition,
    SourcePosition,
)

from .io import nexus


# See https://github.com/scipp/essreduce/issues/98
def load_nexus_source_from_moderator(
    location: NeXusComponentLocationSpec[snx.NXsource, RunType],
) -> NeXusSource[RunType]:
    """Load a NeXus moderator as a source."""
    from ess.reduce.nexus import load_component

    return NeXusSource[RunType](load_component(location, nx_class=snx.NXmoderator))


def get_calibrated_detector_bifrost(
    detector: NeXusDetector[RunType],
    *,
    offset: DetectorPositionOffset[RunType],
    source_position: SourcePosition[RunType],
    sample_position: SamplePosition[RunType],
    gravity: GravityVector,
) -> CalibratedDetector[RunType]:
    """Extract the data array corresponding to a detector's signal field.

    The returned data array includes coords and masks pertaining directly to the
    signal values array, but not additional information about the detector.
    The data array is reshaped to the logical detector shape.

    Parameters
    ----------
    detector:
        NeXus detector group.
    offset:
        Offset to add to the detector position.
    source_position:
        Position of the neutron source.
    sample_position:
        Position of the sample.
    gravity:
        Gravity vector.

    Returns
    -------
    :
        Detector data.
    """

    from ess.reduce.nexus.types import DetectorBankSizes
    from ess.reduce.nexus.workflow import get_calibrated_detector

    da = get_calibrated_detector(
        detector=detector,
        offset=offset,
        source_position=source_position,
        sample_position=sample_position,
        gravity=gravity,
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
    workflow.insert(load_nexus_source_from_moderator)
    workflow.insert(get_calibrated_detector_bifrost)
    for key, val in default_parameters().items():
        workflow[key] = val
    return workflow
