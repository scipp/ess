# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
import scipp.testing
import scippnexus as snx

from ess import bifrost
from ess.bifrost.data import (
    simulated_elastic_incoherent_with_phonon,
    tof_lookup_table_simulation,
)
from ess.bifrost.live import CutAxis, CutAxis1, CutAxis2, arc_energy, cut
from ess.spectroscopy.types import (
    EnergyQDetector,
    Filename,
    NeXusDetectorName,
    SampleRun,
    TimeOfFlightLookupTableFilename,
    UncertaintyBroadcastMode,
)


@pytest.fixture(scope='module')
def simulation_detector_names() -> list[NeXusDetectorName]:
    with snx.File(simulated_elastic_incoherent_with_phonon()) as f:
        names = list(f['entry']['instrument'][snx.NXdetector].keys())
    return names[:10]  # First 10 detectors form a 5x2 grid (arc=5, channel=2)


@pytest.fixture
def energy_data(
    simulation_detector_names: list[NeXusDetectorName],
) -> EnergyQDetector[SampleRun]:
    workflow = bifrost.BifrostSimulationWorkflow(simulation_detector_names)
    workflow[Filename[SampleRun]] = simulated_elastic_incoherent_with_phonon()
    workflow[TimeOfFlightLookupTableFilename] = tof_lookup_table_simulation()
    workflow[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.drop
    return workflow.compute(EnergyQDetector[SampleRun])


def test_cut_along_q_norm_and_energy_transfer_preserves_counts(
    energy_data: EnergyQDetector[SampleRun],
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

    cut_data = cut(
        energy_data,
        axis_1=CutAxis1(axis_1),
        axis_2=CutAxis2(axis_2),
        arc_energy=arc_energy(),
    )

    # Verify no counts were lost during the cut
    total_counts_before = sc.sum(energy_data.sum())
    total_counts_after = sc.sum(cut_data)
    sc.testing.assert_allclose(total_counts_before, total_counts_after)


def test_cut_along_qx_direction_preserves_counts(
    energy_data: EnergyQDetector[SampleRun],
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

    cut_data = cut(
        energy_data,
        axis_1=CutAxis1(axis_1),
        axis_2=CutAxis2(axis_2),
        arc_energy=arc_energy(),
    )

    # Verify no counts were lost during the cut
    total_counts_before = sc.sum(energy_data.sum())
    total_counts_after = sc.sum(cut_data)
    sc.testing.assert_allclose(total_counts_before, total_counts_after)
