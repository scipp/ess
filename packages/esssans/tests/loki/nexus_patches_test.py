# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
from importlib.resources import files
from pathlib import Path

import pytest
import sciline
import scipp as sc
import scippnexus as snx

from ess.reduce.nexus import GenericNeXusWorkflow, patch
from ess.reduce.nexus.types import (
    EmptyDetector,
    EmptyMonitor,
    Filename,
    IncidentMonitor,
    NeXusDetectorName,
    NeXusFilePatch,
    NeXusName,
    NeXusTransformationChain,
    SampleRun,
)

CARRIAGE_MOTION = '/entry/instrument/detector_carriage/transformations/motion'


@pytest.fixture
def workflow(loki_broken_file: Path) -> sciline.Pipeline:
    wf = GenericNeXusWorkflow(run_types=[SampleRun], monitor_types=[IncidentMonitor])
    wf[Filename[SampleRun]] = loki_broken_file
    wf[NeXusDetectorName] = 'loki_detector_0'
    wf[NeXusName[IncidentMonitor]] = 'beam_monitor_m4'
    return wf


def test_broken_file_cannot_be_loaded_without_patch(
    workflow: sciline.Pipeline,
) -> None:
    with pytest.raises(ValueError, match='time-dependent'):
        workflow.compute(EmptyDetector[SampleRun])


def test_todo_motion_sources_patch_uses_motor_readback(
    workflow: sciline.Pipeline,
) -> None:
    workflow[NeXusFilePatch[SampleRun]] = patch.load_patch(
        files('ess.loki') / 'nexus_patches' / 'todo_motion_sources.toml'
    )
    chain = workflow.compute(NeXusTransformationChain[snx.NXdetector, SampleRun])
    motion = chain.transformations[CARRIAGE_MOTION].value
    assert motion.sizes == {'time': 1}
    assert sc.isclose(motion.data[0], sc.scalar(4999.96164649, unit='mm'))

    detector = workflow.compute(EmptyDetector[SampleRun])
    monitor = workflow.compute(EmptyMonitor[SampleRun, IncidentMonitor])
    assert sc.all(sc.isfinite(detector.coords['position'])).value
    assert sc.all(sc.isfinite(monitor.coords['position'])).value
