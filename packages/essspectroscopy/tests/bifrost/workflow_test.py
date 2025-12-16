# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest
import sciline
import scipp as sc
import scipp.testing
import scippnexus as snx

from ess import bifrost
from ess.bifrost.data import (
    computed_energy_data_simulated_5x2,
    simulated_elastic_incoherent_with_phonon,
    tof_lookup_table_simulation,
)
from ess.spectroscopy.types import (
    EnergyQDetector,
    Filename,
    FrameMonitor3,
    NeXusDetectorName,
    RawDetector,
    RawMonitor,
    SampleRun,
    TimeOfFlightLookupTableFilename,
    UncertaintyBroadcastMode,
    WavelengthMonitor,
)


@pytest.fixture(scope='module')
def simulation_detector_names() -> list[NeXusDetectorName]:
    with snx.File(simulated_elastic_incoherent_with_phonon()) as f:
        names = list(f['entry']['instrument'][snx.NXdetector].keys())
    return names[:10]  # First 10 detectors form a 5x2 grid (arc=5, channel=2)


@pytest.fixture
def workflow(simulation_detector_names: list[NeXusDetectorName]) -> sciline.Pipeline:
    workflow = bifrost.BifrostSimulationWorkflow(simulation_detector_names)
    workflow[Filename[SampleRun]] = simulated_elastic_incoherent_with_phonon()
    workflow[TimeOfFlightLookupTableFilename] = tof_lookup_table_simulation()
    workflow[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.drop
    return workflow


def test_simulation_workflow_can_load_detector() -> None:
    workflow = bifrost.BifrostSimulationWorkflow(
        [NeXusDetectorName("125_channel_1_1_triplet")]
    )
    workflow[Filename[SampleRun]] = simulated_elastic_incoherent_with_phonon()
    results = sciline.compute_mapped(workflow, RawDetector[SampleRun])
    result = results.iloc[0]

    assert result.bins is not None
    assert set(result.dims) == {'tube', 'length'}
    assert result.sizes['tube'] == 3
    assert 'position' in result.coords


def test_simulation_workflow_can_load_monitor(workflow: sciline.Pipeline) -> None:
    result = workflow.compute(RawMonitor[SampleRun, FrameMonitor3])

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
        'a4': 1,
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
    monitor = workflow.compute(WavelengthMonitor[SampleRun, FrameMonitor3])
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
    assert not energy_data.bins.masks

    assert set(expected.coords.keys()).issubset(set(energy_data.coords.keys()))
    for name in expected.coords.keys():
        sc.testing.assert_allclose(energy_data.coords[name], expected.coords[name])

    assert energy_data.bins.coords.keys() == expected.bins.coords.keys()
    for name in energy_data.bins.coords.keys():
        sc.testing.assert_allclose(
            energy_data.bins.coords[name], expected.bins.coords[name]
        )

    sc.testing.assert_allclose(energy_data.bins.data, expected.bins.data)
