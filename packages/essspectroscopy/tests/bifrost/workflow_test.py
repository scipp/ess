# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest
import sciline
import scipp as sc
import scipp.testing
import scippnexus as snx

from ess import bifrost
from ess.bifrost.data import (
    computed_energy_data_simulated,
    simulated_elastic_incoherent_with_phonon,
    tof_lookup_table_simulation,
)
from ess.bifrost.types import FrameMonitor3
from ess.spectroscopy.types import (
    DetectorData,
    EnergyData,
    Filename,
    MonitorData,
    NeXusDetectorName,
    SampleRun,
    TimeOfFlightLookupTable,
    WavelengthMonitor,
)


@pytest.fixture(scope='module')
def simulation_detector_names() -> list[NeXusDetectorName]:
    with snx.File(simulated_elastic_incoherent_with_phonon()) as f:
        names = list(f['entry']['instrument'][snx.NXdetector].keys())
    return names[:5]  # These should be enough to test the workflow.


@pytest.fixture
def workflow(simulation_detector_names: list[NeXusDetectorName]) -> sciline.Pipeline:
    workflow = bifrost.BifrostSimulationWorkflow(simulation_detector_names)
    workflow[Filename[SampleRun]] = simulated_elastic_incoherent_with_phonon()
    workflow[TimeOfFlightLookupTable] = sc.io.load_hdf5(tof_lookup_table_simulation())
    return workflow


def test_simulation_workflow_can_load_detector() -> None:
    workflow = bifrost.BifrostSimulationWorkflow(
        [NeXusDetectorName("125_channel_1_1_triplet")]
    )
    workflow[Filename[SampleRun]] = simulated_elastic_incoherent_with_phonon()
    results = sciline.compute_mapped(workflow, DetectorData[SampleRun])
    result = results.iloc[0]

    assert result.bins is not None
    assert set(result.dims) == {'tube', 'length'}
    assert result.sizes['tube'] == 3
    assert 'position' in result.coords
    assert 'sample_position' in result.coords
    assert 'source_position' in result.coords


def test_simulation_workflow_can_load_monitor(workflow: sciline.Pipeline) -> None:
    result = workflow.compute(MonitorData[SampleRun, FrameMonitor3])

    assert result.bins is None
    assert 'position' in result.coords
    assert 'sample_position' not in result.coords
    assert 'source_position' in result.coords


def test_simulation_workflow_can_compute_energy_data(
    workflow: sciline.Pipeline,
) -> None:
    energy_data = workflow.compute(EnergyData[SampleRun])

    assert energy_data.sizes == {
        'triplet': 5,
        'tube': 3,
        'length': 100,
        'a3': 180,
        'a4': 1,
    }
    expected_coords = {'a3', 'a4', 'detector_number'}
    assert expected_coords.issubset(energy_data.coords)
    expected_event_coords = {
        'incident_wavelength',
        'energy_transfer',
        'lab_momentum_transfer',
        'sample_table_momentum_transfer',
    }
    assert expected_event_coords.issubset(energy_data.bins.coords)

    # Check that conversions do not raise, i.e., units have expected dimensions.
    energy_data.coords['a3'].to(unit='rad')
    energy_data.coords['a4'].to(unit='rad')
    energy_data.bins.coords['energy_transfer'].to(unit='meV')
    energy_data.bins.coords['lab_momentum_transfer'].to(unit='1/Å')
    energy_data.bins.coords['sample_table_momentum_transfer'].to(unit='1/Å')


def test_simulation_workflow_can_compute_wavelength_monitor(
    workflow: sciline.Pipeline,
) -> None:
    monitor = workflow.compute(WavelengthMonitor[SampleRun, FrameMonitor3])
    assert set(monitor.dims) == {'time', 'wavelength'}
    expected_coords = {'position', 'wavelength', 'time'}
    assert expected_coords.issubset(monitor.coords)
    assert monitor.bins is None


def test_simulation_workflow_produces_the_same_data_as_before(
    workflow: sciline.Pipeline,
) -> None:
    energy_data = workflow.compute(EnergyData[SampleRun])
    expected = sc.io.load_hdf5(computed_energy_data_simulated())
    sc.testing.assert_allclose(energy_data, expected)
