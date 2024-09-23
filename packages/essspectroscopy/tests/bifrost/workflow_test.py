# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from ess import bifrost
from ess.spectroscopy.types import (
    DetectorData,
    Filename,
    Monitor3,
    MonitorData,
    NeXusDetectorName,
    SampleRun,
)


def test_simulation_workflow_can_load_detector() -> None:
    workflow = bifrost.BifrostSimulationWorkflow()
    workflow[Filename[SampleRun]] = (
        bifrost.data.simulated_elastic_incoherent_with_phonon()
    )
    workflow[NeXusDetectorName] = "125_channel_1_1_triplet"
    result = workflow.compute(DetectorData[SampleRun])

    assert result.bins is not None
    assert set(result.dims) == {'tube', 'length'}
    assert result.sizes['tube'] == 3
    assert 'position' in result.coords
    assert 'sample_position' in result.coords
    assert 'source_position' in result.coords


def test_simulation_workflow_can_load_monitor() -> None:
    workflow = bifrost.BifrostSimulationWorkflow()
    workflow[Filename[SampleRun]] = (
        bifrost.data.simulated_elastic_incoherent_with_phonon()
    )
    result = workflow.compute(MonitorData[SampleRun, Monitor3])

    assert result.bins is None
    assert 'position' in result.coords
    assert 'sample_position' not in result.coords
    assert 'source_position' in result.coords
