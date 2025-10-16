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
from ess.bifrost.live import BifrostQCutWorkflow, CutAxis, CutAxis1, CutAxis2, CutData
from ess.spectroscopy.types import (
    DetectorData,
    EnergyData,
    Filename,
    FrameMonitor3,
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
        'arc': 5,
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

    assert not energy_data.masks
    assert not energy_data.bins.masks

    # Handle transition from 'triplet' to 'arc' dimension
    if 'triplet' in expected.dims and 'arc' in energy_data.dims:
        expected = expected.rename_dims(triplet='arc')

    assert energy_data.coords.keys() == expected.coords.keys()
    for name in energy_data.coords.keys():
        sc.testing.assert_allclose(energy_data.coords[name], expected.coords[name])

    assert energy_data.bins.coords.keys() == expected.bins.coords.keys()
    for name in energy_data.bins.coords.keys():
        sc.testing.assert_allclose(
            energy_data.bins.coords[name], expected.bins.coords[name]
        )

    sc.testing.assert_allclose(energy_data.bins.data, expected.bins.data)


class TestBifrostQCutWorkflow:
    @pytest.fixture
    def qcut_workflow(
        self, simulation_detector_names: list[NeXusDetectorName]
    ) -> sciline.Pipeline:
        workflow = BifrostQCutWorkflow(simulation_detector_names)
        workflow[Filename[SampleRun]] = simulated_elastic_incoherent_with_phonon()
        workflow[TimeOfFlightLookupTable] = sc.io.load_hdf5(
            tof_lookup_table_simulation()
        )
        return workflow

    def test_cut_along_q_norm_and_energy_transfer(
        self, qcut_workflow: sciline.Pipeline
    ) -> None:
        # Define cut axes for |Q| and energy transfer
        axis_1 = CutAxis(
            output='|Q|',
            fn=lambda sample_table_momentum_transfer: sc.norm(
                sample_table_momentum_transfer
            ),
            bins=sc.linspace(dim='|Q|', start=0.0, stop=3.0, num=50, unit='1/angstrom'),
        )
        axis_2 = CutAxis(
            output='E',
            fn=lambda energy_transfer: energy_transfer,
            bins=sc.linspace(dim='E', start=-10.0, stop=10.0, num=50, unit='meV'),
        )

        qcut_workflow[CutAxis1] = axis_1
        qcut_workflow[CutAxis2] = axis_2

        # Compute both cut data and energy data to compare total counts
        cut_data = qcut_workflow.compute(CutData[SampleRun])
        energy_data = qcut_workflow.compute(EnergyData[SampleRun])

        # Verify the structure of the result (now 3-D with arc dimension)
        assert cut_data.bins is None  # Should be histogrammed
        assert set(cut_data.dims) == {'arc', '|Q|', 'E'}
        assert cut_data.sizes['arc'] == 5
        assert cut_data.sizes['|Q|'] == 49  # num bins - 1
        assert cut_data.sizes['E'] == 49
        assert 'arc' in cut_data.coords
        assert '|Q|' in cut_data.coords
        assert 'E' in cut_data.coords

        # Check that coordinates have expected units
        cut_data.coords['arc'].to(unit='meV')
        cut_data.coords['|Q|'].to(unit='1/angstrom')
        cut_data.coords['E'].to(unit='meV')

        # Verify no counts were lost during the cut
        total_counts_before = sc.sum(energy_data.bins.size()).value
        total_counts_after = sc.sum(cut_data).value
        assert total_counts_before == total_counts_after

    def test_cut_along_qx_direction_and_energy_transfer(
        self, qcut_workflow: sciline.Pipeline
    ) -> None:
        # Test cutting along a specific Q direction (Qx)
        axis_1 = CutAxis.from_q_vector(
            output='Qx',
            vec=sc.vector([1, 0, 0]),
            bins=sc.linspace(dim='Qx', start=-2.0, stop=2.0, num=40, unit='1/angstrom'),
        )
        axis_2 = CutAxis(
            output='E',
            fn=lambda energy_transfer: energy_transfer,
            bins=sc.linspace(dim='E', start=-5.0, stop=5.0, num=30, unit='meV'),
        )

        qcut_workflow[CutAxis1] = axis_1
        qcut_workflow[CutAxis2] = axis_2

        # Compute both cut data and energy data to compare total counts
        cut_data = qcut_workflow.compute(CutData[SampleRun])
        energy_data = qcut_workflow.compute(EnergyData[SampleRun])

        # Verify the result structure (now 3-D with arc dimension)
        assert cut_data.bins is None
        assert set(cut_data.dims) == {'arc', 'Qx', 'E'}
        assert cut_data.sizes['arc'] == 5
        assert cut_data.sizes['Qx'] == 39
        assert cut_data.sizes['E'] == 29
        assert 'arc' in cut_data.coords
        assert 'Qx' in cut_data.coords
        assert 'E' in cut_data.coords

        # Verify no counts were lost during the cut
        total_counts_before = sc.sum(energy_data.bins.size()).value
        total_counts_after = sc.sum(cut_data).value
        assert total_counts_before == total_counts_after

    def test_cut_preserves_arc_dimension(self, qcut_workflow: sciline.Pipeline) -> None:
        # Test that cut preserves the arc dimension (renamed from triplet)
        axis_1 = CutAxis(
            output='|Q|',
            fn=lambda sample_table_momentum_transfer: sc.norm(
                sample_table_momentum_transfer
            ),
            bins=sc.linspace(dim='|Q|', start=0.0, stop=3.0, num=50, unit='1/angstrom'),
        )
        axis_2 = CutAxis(
            output='E',
            fn=lambda energy_transfer: energy_transfer,
            bins=sc.linspace(dim='E', start=-10.0, stop=10.0, num=50, unit='meV'),
        )

        qcut_workflow[CutAxis1] = axis_1
        qcut_workflow[CutAxis2] = axis_2

        cut_data = qcut_workflow.compute(CutData[SampleRun])

        # Verify the result is 3-D with arc dimension preserved
        assert cut_data.bins is None
        assert set(cut_data.dims) == {'arc', '|Q|', 'E'}
        assert cut_data.sizes['arc'] == 5
        assert cut_data.sizes['|Q|'] == 49
        assert cut_data.sizes['E'] == 49

        # Verify arc coordinate exists with correct values
        assert 'arc' in cut_data.coords
        expected_arc_energies = sc.array(
            dims=['arc'], values=[2.7, 3.2, 3.8, 4.4, 5.0], unit='meV'
        )
        sc.testing.assert_allclose(cut_data.coords['arc'], expected_arc_energies)
