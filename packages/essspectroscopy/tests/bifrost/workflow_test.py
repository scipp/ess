# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import random

import numpy as np
import pytest
import sciline
import scipp as sc
import scipp.testing
import scippnexus as snx
from ess import bifrost
from ess.bifrost.data import (
    coda_test_run_2026_10,
    computed_energy_data_simulated_5x2,
    lookup_table_simulation,
    simulated_elastic_incoherent_with_phonon,
)
from ess.bifrost.types import NeXusDetectorNames
from ess.spectroscopy.types import (
    Analyzers,
    DataAtSample,
    ElasticMonitor,
    EnergyQDetector,
    Filename,
    LookupTableFilename,
    LookupTableRelativeErrorThreshold,
    NeXusDetectorName,
    NormalizationMonitor,
    OverlapMonitor,
    PSCMonitor,
    RawDetector,
    RawMonitor,
    SampleRun,
    TransformationOverrides,
    UncertaintyBroadcastMode,
    WavelengthMonitor,
)


@pytest.fixture(scope='module')
def detector_names() -> list[str]:
    # Limit the number of detectors to speed up tests.
    # First 10 detectors form a 5x2 grid (arc=5, channel=2)
    return [
        f'channel_{channel}_{arc}_triplet'
        for arc in range(1, 6)
        for channel in range(1, 3)
    ]


@pytest.fixture
def workflow(detector_names: list[str]) -> sciline.Pipeline:
    workflow = bifrost.BifrostSimulationWorkflow()
    workflow[NeXusDetectorNames] = detector_names
    workflow[Filename[SampleRun]] = simulated_elastic_incoherent_with_phonon()
    workflow[LookupTableFilename] = lookup_table_simulation()
    workflow[LookupTableRelativeErrorThreshold] = {
        'detector': np.inf,
        'normalization_monitor': np.inf,
    }
    workflow[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.drop
    return workflow


def test_simulation_workflow_can_load_single_detector() -> None:
    workflow = bifrost.BifrostSimulationWorkflow()
    workflow[NeXusDetectorNames] = [NeXusDetectorName("channel_3_1_triplet")]
    workflow[Filename[SampleRun]] = simulated_elastic_incoherent_with_phonon()
    result = workflow.compute(RawDetector[SampleRun])

    assert result.bins is not None
    assert set(result.dims) == {'arc', 'channel', 'tube', 'length', 'time'}
    assert result.sizes['tube'] == 3
    assert 'position' in result.coords


@pytest.mark.parametrize('shuffle', [False, True])
def test_simulation_workflow_can_load_multiple_detectors_rectangular(
    shuffle: bool,
) -> None:
    # Load a subset of detectors that form a rectangular region in the arc-channel space
    # These detectors can be folded to (arc, channel) dims.
    detector_names = [
        f'channel_{channel}_{arc}_triplet'
        for arc in range(2, 5)
        for channel in range(4, 10)
    ]
    if shuffle:
        # The order does not matter.
        random.shuffle(detector_names)

    workflow = bifrost.BifrostSimulationWorkflow()
    workflow[NeXusDetectorNames] = detector_names
    workflow[Filename[SampleRun]] = simulated_elastic_incoherent_with_phonon()
    result = workflow.compute(RawDetector[SampleRun])

    assert result.bins is not None
    assert set(result.dims) == {'arc', 'channel', 'tube', 'length', 'time'}
    assert result.sizes['arc'] == 3
    assert result.sizes['channel'] == 6
    assert result.sizes['tube'] == 3
    assert 'position' in result.coords


def test_simulation_workflow_can_load_multiple_detectors_non_rectangular() -> None:
    # Load a subset of detectors that do not form a rectangular region in the
    # arc-channel space. These detectors can *not* be folded to (arc, channel) dims.
    detector_names = [
        'channel_3_1_triplet',
        'channel_3_2_triplet',
        'channel_4_1_triplet',
        'channel_4_2_triplet',
        'channel_5_2_triplet',
        'channel_6_3_triplet',
    ]

    workflow = bifrost.BifrostSimulationWorkflow()
    workflow[NeXusDetectorNames] = detector_names
    workflow[Filename[SampleRun]] = simulated_elastic_incoherent_with_phonon()
    result = workflow.compute(RawDetector[SampleRun])

    assert result.bins is not None
    assert set(result.dims) == {'triplet', 'tube', 'length', 'time'}
    assert result.sizes['triplet'] == len(detector_names)
    assert result.sizes['tube'] == 3
    assert 'position' in result.coords


def test_simulation_workflow_can_load_monitor(workflow: sciline.Pipeline) -> None:
    result = workflow.compute(RawMonitor[SampleRun, NormalizationMonitor])

    assert result.bins is None
    assert 'position' in result.coords


def test_simulation_workflow_can_compute_energy_data(
    workflow: sciline.Pipeline,
) -> None:
    energy_data = workflow.compute(EnergyQDetector[SampleRun])

    assert energy_data.sizes == {
        'arc': 5,
        'channel': 2,
        'tube': 3,
        'length': 100,
        'a3': 180,
        'a4': 2,
    }
    expected_coords = {'a3', 'a4', 'detector_number'}
    assert expected_coords.issubset(energy_data.coords)
    expected_event_coords = {
        'energy_transfer',
        'sample_table_momentum_transfer',
    }
    assert expected_event_coords.issubset(energy_data.bins.coords)

    # Check that conversions do not raise, i.e., units have expected dimensions.
    energy_data.coords['a3'].to(unit='rad')
    energy_data.coords['a4'].to(unit='rad')
    energy_data.bins.coords['energy_transfer'].to(unit='meV')
    energy_data.bins.coords['sample_table_momentum_transfer'].to(unit='1/Å')


def test_simulation_workflow_can_compute_wavelength_monitor(
    workflow: sciline.Pipeline,
) -> None:
    monitor = workflow.compute(WavelengthMonitor[SampleRun, NormalizationMonitor])
    assert set(monitor.dims) == {'time', 'incident_wavelength'}
    expected_coords = {'position', 'incident_wavelength', 'time'}
    assert expected_coords.issubset(monitor.coords)
    assert monitor.bins is None


def test_simulation_workflow_produces_the_same_data_as_before(
    workflow: sciline.Pipeline,
) -> None:
    energy_data = workflow.compute(EnergyQDetector[SampleRun])
    expected = sc.io.load_hdf5(computed_energy_data_simulated_5x2())

    assert not energy_data.masks

    assert set(expected.coords.keys()).issubset(set(energy_data.coords.keys()))
    for name in expected.coords.keys():
        sc.testing.assert_allclose(
            energy_data.coords[name],
            expected.coords[name],
            atol=(
                sc.scalar(1e-15, unit=expected.coords[name].unit)
                # TODO: When scipp supports unit=None in .to(), remove this case.
                if expected.coords[name].unit is not None
                else None
            ),
        )

    assert energy_data.bins.coords.keys() == expected.bins.coords.keys()
    for name in energy_data.bins.coords.keys():
        sc.testing.assert_allclose(
            energy_data.bins.coords[name],
            expected.bins.coords[name],
            atol=(
                sc.scalar(1e-15, unit=expected.bins.coords[name].unit)
                # TODO: When scipp supports unit=None in .to(), remove this case.
                if expected.bins.coords[name].unit is not None
                else None
            ),
        )

    sc.testing.assert_allclose(
        energy_data.bins.data,
        expected.bins.data,
        atol=sc.scalar(1e-15, unit=expected.unit),
    )


@pytest.fixture
def coda_workflow(detector_names: list[str]) -> sciline.Pipeline:
    workflow = bifrost.BifrostWorkflow()
    workflow[NeXusDetectorNames] = detector_names
    workflow[Filename[SampleRun]] = coda_test_run_2026_10()
    # The table does not match the choppers of this run but these tests only check
    # that the file can be loaded.
    workflow[LookupTableFilename] = lookup_table_simulation()
    workflow[LookupTableRelativeErrorThreshold] = {'detector': np.inf}
    return workflow


def test_tank_angle_override_rotates_detectors_and_analyzers_and_sets_a4(
    coda_workflow: sciline.Pipeline,
) -> None:
    tank_angle = '/entry/instrument/detector_tank/transformations/tank_angle'
    # The angle in the file is 0, so the overridden positions are the original ones
    # rotated by the override, around the vertical axis through the sample.
    original_detector = coda_workflow.compute(RawDetector[SampleRun])
    original_analyzers = coda_workflow.compute(Analyzers[SampleRun])

    angle = sc.scalar(30.0, unit='deg')
    rotation = sc.spatial.rotations_from_rotvecs(angle * sc.vector([0.0, 1.0, 0.0]))
    # The tank carries all of these, so each needs the override.
    for component in (snx.NXdetector, snx.NXcrystal, ElasticMonitor):
        coda_workflow[TransformationOverrides[component, SampleRun]] = {
            tank_angle: angle
        }

    atol = sc.scalar(1e-9, unit='m')
    sc.testing.assert_allclose(
        coda_workflow.compute(RawDetector[SampleRun]).coords['position'],
        rotation * original_detector.coords['position'],
        atol=atol,
    )
    for name, analyzer in coda_workflow.compute(Analyzers[SampleRun]).items():
        sc.testing.assert_allclose(
            analyzer['position'].position,
            rotation * original_analyzers[name]['position'].position,
            atol=atol,
        )
    data = coda_workflow.compute(DataAtSample[SampleRun])
    sc.testing.assert_identical(data.coords['a4'], sc.concat([angle], 'a4'))


def test_workflow_can_load_detector_with_analyzers_and_angles_from_coda_file(
    coda_workflow: sciline.Pipeline,
) -> None:
    data = coda_workflow.compute(DataAtSample[SampleRun])
    assert data.sizes == {
        'arc': 5,
        'channel': 2,
        'tube': 3,
        'length': 100,
        'a3': 1,
        'a4': 1,
    }
    assert data.bins.size().sum().value > 0
    # Requires the analyzer of each triplet.
    assert data.coords['final_energy'].unit == 'meV'


@pytest.mark.parametrize(
    'monitor_type',
    [ElasticMonitor, NormalizationMonitor, OverlapMonitor, PSCMonitor],
)
def test_workflow_can_load_monitors_from_coda_file(
    coda_workflow: sciline.Pipeline, monitor_type: type
) -> None:
    monitor = coda_workflow.compute(RawMonitor[SampleRun, monitor_type])
    assert 'position' in monitor.coords
