# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from ess import bifrost
from ess.bifrost.data import simulated_elastic_incoherent_with_phonon
from ess.bifrost.types import FrameMonitor3
from ess.spectroscopy.types import (
    Analyzers,
    Choppers,
    DetectorData,
    Filename,
    MonitorData,
    NeXusDetectorName,
    SampleRun,
)


def test_simulation_workflow_can_load_detector() -> None:
    workflow = bifrost.BifrostSimulationWorkflow()
    workflow[Filename[SampleRun]] = simulated_elastic_incoherent_with_phonon()
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
    workflow[Filename[SampleRun]] = simulated_elastic_incoherent_with_phonon()
    result = workflow.compute(MonitorData[SampleRun, FrameMonitor3])

    assert result.bins is None
    assert 'position' in result.coords
    assert 'sample_position' not in result.coords
    assert 'source_position' in result.coords


def test_simulation_workflow_can_load_analyzers() -> None:
    workflow = bifrost.BifrostSimulationWorkflow()
    workflow[Filename[SampleRun]] = simulated_elastic_incoherent_with_phonon()
    analyzers = workflow.compute(Analyzers[SampleRun])

    assert len(analyzers) == 45
    first = next(iter(analyzers.values()))
    assert 'position' in first
    assert 'd_spacing' in first


def test_simulation_workflow_can_load_choppers() -> None:
    workflow = bifrost.BifrostSimulationWorkflow()
    workflow[Filename[SampleRun]] = simulated_elastic_incoherent_with_phonon()
    choppers = workflow.compute(Choppers[SampleRun])

    assert choppers.keys() == {
        '005_PulseShapingChopper',
        '006_PulseShapingChopper2',
        '019_FOC1',
        '048_FOC2',
        '095_BWC1',
        '096_BWC2',
    }
    first = next(iter(choppers.values()))
    assert 'position' in first
    assert 'rotation_speed' in first
    assert first['slit_edges'].shape == (2,)
